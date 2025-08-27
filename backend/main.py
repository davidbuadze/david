# main.py
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import os
import io
import uuid
import asyncio
from typing import Optional, List, Dict, Any
import datetime

import numpy as np
# Firebase Admin SDK
import firebase_admin
import logging
from firebase_admin import credentials, auth, firestore
from fastapi.middleware.cors import CORSMiddleware
from google.api_core import exceptions as google_exceptions
from google.cloud.firestore_v1.base_timestamp import Timestamp

# Google Cloud Libraries for LLM and Storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import storage

# Libraries for document parsing
from pypdf import PdfReader
from docx import Document

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- Глобальные переменные для Firebase и GCP ---
# Получаем ID проекта и имя бакета из переменных окружения
# Если они не установлены, это указывает на проблему с развертыванием
# и приложение не сможет функционировать корректно.
# Убедитесь, что эти переменные окружения установлены в Cloud Run!
CP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_BOOKS_BUCKET = os.environ.get("GCS_BOOKS_BUCKET")
GCS_USER_DATA_BUCKET = os.environ.get("GCS_USER_DATA_BUCKET")
GCS_HOSTING_BUCKET = os.environ.get("GCS_HOSTING_BUCKET")

# Проверяем, что необходимые переменные окружения установлены
if not GCP_PROJECT_ID:
    raise ValueError("Ошибка: Переменная окружения GCP_PROJECT_ID не установлена.")
if not GCS_BUCKET_NAME:
    raise ValueError("Ошибка: Переменная окружения GCS_BUCKET_NAME не установлена.")

# --- Инициализация Firebase Admin SDK и Firestore ---
app = FastAPI(
    title="Обучающее Приложение Backend API",
    description="API для управления пользователями, обучающими планами, LLM-агентом и RAG.",
    version="0.1.0",
)

db = None
try:
    if not firebase_admin._apps: # Проверяем, не инициализировано ли уже приложение
        firebase_admin.initialize_app(credentials.ApplicationDefault())
    db = firestore.client()
    print("Firebase Admin SDK успешно инициализирован.")
except Exception as e:
    print(f"Ошибка инициализации Firebase Admin SDK: {e}")
    # Это критическая ошибка, без которой приложение не может работать.
    # Лучше "упасть" сразу с понятным сообщением.
    raise RuntimeError(f"Не удалось инициализировать Firebase Admin SDK: {e}") from e

# --- Инициализация Vertex AI ---
llm_model = None
embedding_model = None
try:
    if GCP_PROJECT_ID and GCP_LOCATION:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        llm_model = GenerativeModel("gemini-2.5-flash-preview-05-20")
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print(f"Vertex AI initialized for project {GCP_PROJECT_ID} in {GCP_LOCATION}")
    else:
        print("Пропуск инициализации Vertex AI: GCP_PROJECT_ID или GCP_LOCATION не установлены.")
except Exception as e:
    print(f"Предупреждение: Не удалось инициализировать Vertex AI: {e}. Функции LLM будут недоступны.")

# --- Инициализация клиента Cloud Storage ---
storage_client = storage.Client()
print("Google Cloud Storage успешно инициализирован.")
storage_client = None
try:
    if GCP_PROJECT_ID:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        print("Cloud Storage Client успешно инициализирован.")
    else:
        print("Пропуск инициализации Cloud Storage Client: GCP_PROJECT_ID не установлен.")
except Exception as e:
    print(f"Предупреждение: Не удалось инициализировать Cloud Storage Client: {e}. Функции хранилища будут недоступны.")

@app.on_event("startup")
async def startup_event():
    print("Запуск приложения: Выполнение инициализации коллекций...")
    await initialize_collections()
    print("Запуск приложения: Инициализация коллекций завершена.")

# --- Dependency для получения ID пользователя из токена Firebase Auth ---
async def get_current_user(request: Request):
    """
    Проверяет Firebase ID Token из заголовка Authorization.
    Возвращает UID пользователя.
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Отсутствует заголовок авторизации")

    token = auth_header.split('Bearer ')[1] if 'Bearer ' in auth_header else None
    if not token:
        raise HTTPException(status_code=401, detail="Неверный формат токена")

    try:
        decoded_token = auth.verify_id_token(token)
        request.state.user_id = decoded_token['uid']
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Недействительный токен: {e}")

# --- Настройка CORS Middleware ---

# Лучшая практика: загружать разрешенные источники из переменной окружения.
# Это позволяет легко изменять их для разных сред (разработка, продакшн)
# без изменения кода.
# Пример переменной в Cloud Run: "https://ailbee.web.app,https://ailbee.firebaseapp.com"
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS")

if CORS_ALLOWED_ORIGINS:
    origins = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",")]
else:
    # Фоллбэк на жестко закодированный список для локальной разработки,
    # если переменная окружения не установлена.
    # В продакшене настоятельно рекомендуется использовать переменную окружения.
    print("Предупреждение: Переменная окружения CORS_ALLOWED_ORIGINS не установлена. Используется список по умолчанию для разработки.")
    origins = [
        "https://ailbee.web.app",
        "https://ailbee.firebaseapp.com",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:8000",
    ]

print(f"CORS настроен для следующих источников: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS").split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных для запросов ---

class UserProfileCreate(BaseModel):
    userId: str
    email: str
    displayName: str
    age: int
    preferredLanguage: str
    initialGoal: str

class UserProfileUpdate(BaseModel):
    displayName: Optional[str] = None
    age: Optional[int] = None
    preferredLanguage: Optional[str] = None

class LearningPlanGenerateRequest(BaseModel):
    subject: str
    planType: str
    language: Optional[str] = None
    targetGoal: Optional[str] = None
    userPreferences: Optional[dict] = None

class AgentChatMessage(BaseModel):
    message: str
    conversationId: Optional[str] = None
    document_ids: Optional[List[str]] = None
    context: Optional[dict] = None

class GamificationCompleteTask(BaseModel):
    taskId: str
    taskType: str
    pointsEarned: Optional[int] = None
    achievementId: Optional[str] = None
    details: Optional[dict] = None

class FileUploadRequest(BaseModel):
    fileName: str
    fileType: str
    contentType: str
    fileSize: Optional[int] = None

class BillingPurchaseWebhook(BaseModel):
    version: str
    packageName: str
    eventTimeMillis: str
    subscriptionNotification: Optional[dict] = None
    testNotification: Optional[dict] = None

class FrontendSaveDataPayload(BaseModel):
    userId: str
    data: Dict[str, Any]

class FirestoreDocument(BaseModel):
    id: str
    data: Dict[str, Any]

class BackendDataResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None

class LibraryBookCreate(BaseModel):
    title: str
    author: str
    publisher: Optional[str] = None
    publicationYear: Optional[int] = None
    description: Optional[str] = None
    language: str = "ru"
    thumbnailUrl: Optional[str] = None
    contentStoragePath: str
    educationLevel: str = Field(..., description="Уровень образования (primary, middle, high, university, professional)")
    subject: str = Field(..., description="Предмет (Математика, Биология, История, Физика, Литература, Химия, Информатика)")
    tags: Optional[List[str]] = None

class SummarizeTextRequest(BaseModel):
    text: str

# --- Вспомогательная функция для преобразования Firestore Timestamp в строки ---
def convert_timestamps_to_strings(data: Any) -> Any:
    """
    Рекурсивно преобразует объекты Firestore Timestamp и datetime в строки ISO 8601
    внутри словарей и списков.
    """
    if isinstance(data, FirestoreTimestamp):
        return data.isoformat()
    elif isinstance(data, datetime.datetime):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: convert_timestamps_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps_to_strings(elem) for elem in data]
    else:
        return data

# --- Глобальные обработчики исключений ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
 """
 Handles FastAPI HTTPExceptions and returns a standardized JSON response.
 """
 logging.error(f"HTTP Exception occurred: {exc.status_code} - {exc.detail}")
 return JSONResponse(
 status_code=exc.status_code,
 content={"status": "error", "message": exc.detail}
 )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
 """
 Handles all other exceptions and returns a generic 500 error response.
 """
 logging.error(f"An unhandled exception occurred: {exc}", exc_info=True) # Log traceback
 return JSONResponse(status_code=500, content={"status": "error", "message": "Произошла внутренняя ошибка сервера."})

# --- API для загрузки пользовательских данных ---
class UploadResponse(BaseModel):
    status: str
    message: str
    file_url: str

@app.post("/upload_user_data", response_model=UploadResponse)
async def upload_user_data(user_data_file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """
    Загружает файл, предоставленный пользователем, в бакет 'ailbee-user-data'.
    Файл сохраняется в директорию, специфичную для пользователя.
    """
    print(f"Получен запрос на загрузку пользовательских данных от {user_id}")
    
    try:
        bucket_name = GCS_USER_DATA_BUCKET or "ailbee-user-data"
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось получить доступ к бакету: {str(e)}")

    file_extension = user_data_file.filename.split('.')[-1]
    file_path = f"users/{user_id}/{uuid.uuid4()}.{file_extension}"
    blob = bucket.blob(file_path)

    try:
        file_bytes = await user_data_file.read()
        await asyncio.to_thread(blob.upload_from_string, file_bytes, content_type=user_data_file.content_type)
        file_url = blob.public_url
        print(f"Файл '{user_data_file.filename}' успешно загружен в бакет {bucket_name} по пути {file_path}")

        user_files_ref = db.collection('users').document(user_id).collection('files')
        await asyncio.to_thread(user_files_ref.add, {
            "file_name": user_data_file.filename,
            "file_url": file_url,
            "content_type": user_data_file.content_type,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        print(f"Метаданные файла сохранены в Firestore для пользователя {user_id}")

        return {"status": "success", "message": "Файл успешно загружен.", "file_url": file_url}

    except Exception as e:
        print(f"Ошибка при загрузке пользовательских данных: {e}")
        raise HTTPException(status_code=500, detail=f"Произошла ошибка при загрузке файла: {str(e)}")

# --- API для загрузки изображения профиля ---
@app.post("/upload_profile_picture", response_model=UploadResponse)
async def upload_profile_picture(
    profile_picture: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Загружает изображение профиля в бакет 'ailbee-user-data'
    и сохраняет ссылку на него в Firestore.
    """
    print(f"Получен запрос на загрузку изображения профиля от пользователя: {user_id}")

    try:
        bucket_name = GCS_USER_DATA_BUCKET or "ailbee-user-data"
        bucket = storage_client.bucket(bucket_name)
        
        file_extension = profile_picture.filename.split('.')[-1]
        file_path = f"profiles/{user_id}.{file_extension}"
        blob = bucket.blob(file_path)

        file_bytes = await profile_picture.read()
        await asyncio.to_thread(blob.upload_from_string, file_bytes, content_type=profile_picture.content_type)
        
        file_url = blob.public_url
        print(f"Изображение профиля успешно загружено по пути: {file_path}")

        user_doc_ref = db.collection('users').document(user_id)
        await asyncio.to_thread(user_doc_ref.set, {"profile_picture_url": file_url}, merge=True)
        print(f"Ссылка на изображение профиля успешно сохранена в Firestore для пользователя {user_id}")

        return {"status": "success", "message": "Изображение профиля успешно загружено.", "file_url": file_url}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Ошибка при загрузке изображения профиля: {e}")
        raise HTTPException(status_code=500, detail="Произошла ошибка при загрузке файла.")

# --- Вспомогательная функция для генерации стандартизированных ответов об ошибках ---
def generate_error_response(status_code: int, detail: Any) -> JSONResponse:
    """
    Генерирует стандартизированный JSON-ответ об ошибке.
    """
    # В detail может быть строка или словарь (из HTTPException)
    error_message = detail if isinstance(detail, str) else detail.get('detail', 'Произошла неизвестная ошибка.')
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": error_message}
    )

    # --- Инициализация коллекций с начальными данными ---
async def initialize_collections():
    """
    Инициализирует коллекции subjects и educationLevels, если они пусты.
    Эта функция выполняется при запуске приложения.
    """
    if db is None:
        print("Пропуск инициализации коллекций: Firestore DB не инициализирован.")
        return

    subjects_ref = db.collection('subjects')
    education_levels_ref = db.collection('educationLevels')

    try:
        # Проверяем, пуста ли коллекция subjects
        subjects_docs = await asyncio.to_thread(subjects_ref.limit(1).get)
        if not subjects_docs: # Проверяем, пустой ли список документов
            print("Инициализация коллекции 'subjects'...")
            subjects_data = [
                {"id": "Математика", "name": "Математика"},
                {"id": "Биология", "name": "Биология"},
                {"id": "История", "name": "История"},
                {"id": "Физика", "name": "Физика"},
                {"id": "Литература", "name": "Литература"},
                {"id": "Химия", "name": "Химия"},
                {"id": "Информатика", "name": "Информатика"},
                {"id": "Психология", "name": "Психология"}
            ]
            batch = db.batch()
            for subject in subjects_data:
                doc_ref = subjects_ref.document(subject["id"])
                batch.set(doc_ref, subject)
            await asyncio.to_thread(batch.commit)
            print("Коллекция 'subjects' успешно инициализирована.")
        else:
            print("Коллекция 'subjects' уже содержит данные. Пропуск инициализации.")

        # Проверяем, пуста ли коллекция educationLevels
        education_levels_docs = await asyncio.to_thread(education_levels_ref.limit(1).get)
        if not education_levels_docs: # Проверяем, пустой ли список документов
            print("Инициализация коллекции 'educationLevels'...")
            education_levels_data = [
                {"id": "primary", "name": "Начальные классы"},
                {"id": "middle", "name": "Средняя школа"},
                {"id": "high", "name": "Старшие классы"},
                {"id": "university", "name": "Университет"},
                {"id": "professional", "name": "Профессиональный"}
            ]
            batch = db.batch()
            for level in education_levels_data:
                doc_ref = education_levels_ref.document(level["id"])
                batch.set(doc_ref, level)
            await asyncio.to_thread(batch.commit)
            print("Коллекция 'educationLevels' успешно инициализирована.")
        else:
            print("Коллекция 'educationLevels' уже содержит данные. Пропуск инициализации.")

    except Exception as e:
        print(f"Ошибка при инициализации коллекций: {e}")

@app.post("/upload_book")
async def upload_book(book_file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    try:
        # 1. Выбираем бакет для книг
        bucket_name = "ailbee-books"
        bucket = storage_client.bucket(bucket_name)

        # 2. Создаем уникальное имя файла
        file_extension = book_file.filename.split('.')[-1]
        file_path = f"books/{user_id}/{uuid.uuid4()}.{file_extension}"
        blob = bucket.blob(file_path)

        # 3. Загружаем файл в бакет
        file_bytes = await book_file.read()
        await asyncio.to_thread(blob.upload_from_string, file_bytes)

        # 4. Получаем URL загруженного файла
        file_url = blob.public_url
        print(f"Файл успешно загружен в бакет {bucket_name}.")

        # 5. Сохраняем ссылку на файл в коллекции Firestore
        books_collection_ref = db.collection('users').document(user_id).collection('books')
        book_doc_ref = await asyncio.to_thread(books_collection_ref.add, {
            "title": book_file.filename,
            "storage_url": file_url,
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "status": "pending_processing"
        })

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Книга успешно загружена и добавлена в Firestore.",
            "book_id": book_doc_ref[1].id
        })

    except Exception as e:
        print(f"Ошибка загрузки книги: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Произошла ошибка при загрузке книги: {e}"
        })

# --- Имитация векторной базы данных в памяти ---
MOCKED_VECTOR_DB: Dict[str, Dict[str, Any]] = {}
MOCKED_DOC_CHUNKS_MAP: Dict[str, List[str]] = {}

# --- Зависимость для проверки Firebase ID Token ---
async def get_current_user(request: Request):
    """
    Проверяет Firebase ID Token из заголовка Authorization.
    Возвращает UID пользователя.
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        raise HTTPException(status_code=401, detail="Отсутствует заголовок авторизации")

    token = auth_header.split('Bearer ')[1] if 'Bearer ' in auth_header else None
    if not token:
        raise HTTPException(status_code=401, detail="Неверный формат токена")

    try:
        decoded_token = auth.verify_id_token(token)
        request.state.user_id = decoded_token['uid']
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Недействительный токен: {e}")

# --- Вспомогательные функции для RAG ---

async def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Извлекает текст из PDF или DOCX файлов.
    """
    try:
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_content))
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        elif filename.lower().endswith((".doc", ".docx")):
            doc = Document(io.BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else:
            return file_content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Ошибка при извлечении текста из файла {filename}: {e}")
        raise HTTPException(status_code=422, detail=f"Не удалось извлечь текст из файла: {e}")

async def generate_embedding_for_text(text: str) -> List[float]:
    """
    Генерирует векторные эмбеддинги для заданного текста с помощью Vertex AI.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Модель эмбеддингов не инициализирована.")
    try:
        embeddings = embedding_model.embed(text).values
        return embeddings[0]
    except Exception as e:
        print(f"Ошибка при генерации эмбеддингов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации эмбеддингов: {e}")

async def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Разделяет текст на перекрывающиеся фрагменты.
    """
    chunks = []
    if not text:
        return chunks
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def store_document_chunks_in_vector_db(doc_id: str, text_chunks: List[str]):
    """
    Сохраняет фрагменты текста и их эмбеддинги в имитированной векторной БД.
    """
    chunk_ids = []
    for i, chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        embedding = await generate_embedding_for_text(chunk)
        MOCKED_VECTOR_DB[chunk_id] = {"text": chunk, "embedding": embedding}
        chunk_ids.append(chunk_id)
    MOCKED_DOC_CHUNKS_MAP[doc_id] = chunk_ids
    print(f"Документ {doc_id} и его фрагменты сохранены в имитированной векторной БД.")

async def retrieve_relevant_context(query: str, document_ids: List[str]) -> str:
    """
    Извлекает наиболее релевантный контекст из имитированной векторной БД,
    используя косинусное сходство.
    """
    top_k = 5
    print(f"Извлечение {top_k} наиболее релевантных фрагментов для запроса: '{query}' из документов: {document_ids}")

    if not document_ids:
        return "Документы для поиска не указаны."

    try:
        query_embedding = np.array(await generate_embedding_for_text(query))
    except HTTPException as e:
        raise e # Перебрасываем HTTPException из generate_embedding_for_text
    except Exception as e:
        print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Не удалось сгенерировать эмбеддинг для запроса: {e}")

    candidate_chunks = []
    for doc_id in document_ids:
        if doc_id in MOCKED_DOC_CHUNKS_MAP:
            for chunk_id in MOCKED_DOC_CHUNKS_MAP[doc_id]:
                if chunk_id in MOCKED_VECTOR_DB:
                    chunk_data = MOCKED_VECTOR_DB[chunk_id]
                    candidate_chunks.append({
                        "id": chunk_id,
                        "text": chunk_data["text"],
                        "embedding": np.array(chunk_data["embedding"])
                    })
    if not candidate_chunks:
        return "Не удалось найти фрагменты текста в указанных документах."

    # Вычисление косинусного сходства.
    # Эмбеддинги Vertex AI нормализованы, поэтому достаточно скалярного произведения.
    for chunk in candidate_chunks:
        chunk['similarity'] = np.dot(query_embedding, chunk['embedding']) # type: ignore

    # Сортировка по сходству и выбор top_k
    sorted_chunks = sorted(candidate_chunks, key=lambda x: x['similarity'], reverse=True)
    top_chunks = sorted_chunks[:top_k]

    if not top_chunks:
        return "Не удалось найти релевантный контекст в загруженных документах."

    context_texts = [chunk['text'] for chunk in top_chunks]
    full_context = "\n\n---\n\n".join(context_texts)
    MAX_CONTEXT_LENGTH = 7000  # Gemini имеет большое окно контекста
    return full_context[:MAX_CONTEXT_LENGTH]

# --- API Endpoints ---

@app.get("/")
async def root():
    print('Received request for /')
    return {"message": "Welcome to FastAPI Backend!"}

@app.get("/test")
async def test_endpoint():
    print('Received request for /test')
    if db is None:
        return {"message": "Firestore DB failed to initialize!"}
    return {"message": "Hello from Cloud Run! Firestore DB is connected."}

@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(status_code=200)

@app.post("/save_data")
async def save_generic_data(payload: FrontendSaveDataPayload, user_id: str = Depends(get_current_user)):
    if db is None:
        return {"status": "error", "message": "Firestore DB not initialized."}
    data_to_save = payload.data
    data_to_save["timestamp"] = firestore.SERVER_TIMESTAMP

    try:
        doc_ref = db.collection('users').document(user_id).collection('generic_data').document('my_data')
        await asyncio.to_thread(doc_ref.set, data_to_save)
        print(f"Generic data saved for user {user_id} in Firestore...")
        return {"status": "success", "message": "Данные успешно сохранены!", "saved_data": data_to_save}
    except Exception as e:
        print(f"Error saving data to Firestore for user {user_id}: {e}")
        return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}

@app.get("/get_data")
async def get_generic_data(user_id: str = Depends(get_current_user)):
    if db is None:
        return {"status": "error", "message": "Firestore DB not initialized."}
    try:
        doc_ref = db.collection('users').document(user_id).collection('generic_data').document('my_data')
        doc = doc_ref.get()

        if not doc.exists:
            print(f"No generic data found for user {user_id}.")
            return {"status": "success", "message": "Данные не найдены.", "data": None}

        data = doc.to_dict()
        data = convert_timestamps_to_strings(data)
        print(f"Generic data retrieved for user {user_id} from Firestore...")
        return {"status": "success", "message": "Данные успешно получены!", "data": data}
    except Exception as e:
        print(f"Error retrieving data from Firestore for user {user_id}: {e}")
        return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}

# --- API для обработки событий от Eventarc ---
@app.post("/eventarc-handler")
async def handle_eventarc_event(request: Request):
    """
    Обрабатывает входящие события CloudEvents от Eventarc.
    Этот эндпоинт должен быть защищен, и вызывать его должен только Eventarc.
    """
    # CloudEvents отправляются как POST-запросы с телом в формате JSON.
    # Заголовки содержат метаданные события.
    event_type = request.headers.get("ce-type")
    subject = request.headers.get("ce-subject")
    
    print(f"Получено событие Eventarc типа '{event_type}' для субъекта '{subject}'")

    # Полезная нагрузка события находится в теле запроса
    body = await request.json()
    
    # --- Добавьте вашу логику здесь ---
    # Пример для события загрузки файла в Cloud Storage
    if event_type == "google.cloud.storage.object.v1.finalized":
        bucket = body.get("bucket")
        file_name = body.get("name")
        print(f"Новый файл '{file_name}' был загружен в бакет '{bucket}'.")
        # Здесь можно запустить обработку файла, например, индексацию для RAG.

    # Подтвердите получение события, вернув успешный статус-код (2xx).
    return Response(status_code=204) # 204 No Content - хороший выбор, так как ответ не нужен.

# --- API для получения списка книг ---
class LibraryBookCreate(BaseModel):
    title: str
    author: str
    subject: str
    education_level: str
    file_url: str
    description: Optional[str] = None
    
# --- Вспомогательная функция для конвертации Timestamp в строку ---
# Примечание: Эта функция не была предоставлена, но необходима для корректной работы.
# Вам нужно будет реализовать ее, чтобы она соответствовала вашим требованиям.
def convert_timestamps_to_strings(data: dict) -> dict:
    """
    Рекурсивно конвертирует объекты Timestamp в строки ISO 8601.
    """
    for key, value in data.items():
        if isinstance(value, Timestamp):
            data[key] = value.isoformat()
        elif isinstance(value, dict):
            data[key] = convert_timestamps_to_strings(value)
    return data

# --- API для управления книгами библиотеки ---
@app.post("/library/books")
async def create_library_book(
    book_data: LibraryBookCreate,
    user_id: str = Depends(get_current_user)
):
    """
    Добавляет новую книгу в центральную библиотеку.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        book_dict = book_data.model_dump()
        
        book_to_firestore = book_dict.copy()
        book_to_firestore["createdAt"] = firestore.SERVER_TIMESTAMP
        book_to_firestore["updatedAt"] = firestore.SERVER_TIMESTAMP
        book_to_firestore["createdBy"] = user_id

        update_time, doc_ref = await asyncio.to_thread(db.collection('libraryBooks').add, book_to_firestore)
        book_id = doc_ref.id
        print(f"Книга '{book_data.title}' (ID: {book_id}) добавлена в библиотеку.")

        response_book_data = book_dict.copy()
        response_book_data["bookId"] = book_id

        return JSONResponse(status_code=201, content={
            "status": "success",
            "message": "Книга успешно добавлена в библиотеку.",
            "bookId": book_id,
            "book": response_book_data
        })
    except Exception as e:
        print(f"Ошибка при добавлении книги: {e}")
        raise HTTPException(status_code=500, detail="Произошла ошибка при добавлении книги в библиотеку.")

@app.get("/library/books")
async def get_library_books(
    user_id: str = Depends(get_current_user),
    education_level: Optional[str] = Query(None, description="Фильтр по уровню образования (primary, middle, high, university, professional)"),
    subject: Optional[str] = Query(None, description="Фильтр по предмету (Математика, Биология, История, Физика, Литература, Химия, Информатика)")
):
    """
    Получает список книг из центральной библиотеки с возможностью фильтрации.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        print(f"Пользователь {user_id} запросил список книг из библиотеки. Фильтры: уровень={education_level}, предмет={subject}")

        query = db.collection('libraryBooks')

        if education_level:
            query = query.where('education_level', '==', education_level)
        if subject:
            query = query.where('subject', '==', subject)

        books = []
        # Используем to_thread для выполнения блокирующего вызова
        for doc in await asyncio.to_thread(query.stream):
            book_data = doc.to_dict()
            book_data['bookId'] = doc.id
            books.append(convert_timestamps_to_strings(book_data))

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Список книг успешно получен.",
            "books": books
        })
    except google_exceptions.FailedPrecondition as e:
        # Эта ошибка часто возникает при отсутствии необходимых индексов в Firestore
        error_detail = f"Query failed. A required Firestore index is likely missing. Check backend logs for details. Original error: {e}"
        print(error_detail)
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        print(f"Ошибка при получении списка книг: {e}")
        # Возвращаем общую ошибку, чтобы не раскрывать детали реализации
        raise HTTPException(status_code=500, detail="Произошла ошибка при получении списка книг.")

# --- API для LLM-функций ---
@app.post("/llm/summarize_text")
async def summarize_text_with_llm(payload: SummarizeTextRequest, user_id: str = Depends(get_current_user)):
    """
    Суммирует предоставленный текст с помощью Gemini LLM.
    """
    if llm_model is None:
        raise HTTPException(status_code=500, detail="LLM модель не инициализирована.")
    
    if not payload.text:
        raise HTTPException(status_code=400, detail="Текст для суммирования не предоставлен.")

    try:
        prompt = f"Пожалуйста, кратко и четко суммируйте следующий текст:\n\n{payload.text}"
        # Вызов блокирующей функции в отдельном потоке
        response = await asyncio.to_thread(llm_model.generate_content, prompt)
        summarized_text = response.candidates[0].content.parts[0].text
        
        print(f"Текст успешно суммирован для пользователя {user_id}.")
        return JSONResponse(status_code=200, content={
            "status": "success",
            "summary": summarized_text
        })
    except Exception as e:
        print(f"Ошибка при суммировании текста: {e}")
        # Перебрасываем исключение, чтобы его обработал FastAPI
        raise HTTPException(status_code=500, detail="Произошла ошибка при суммировании текста.")

# --- API для управления профилем пользователя ---
@app.post("/users/profile")
async def create_user_profile(profile_data: UserProfileCreate, user_id: str = Depends(get_current_user)):
    """
    Создает/обновляет профиль пользователя в Firestore.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        user_ref = db.collection('users').document(user_id)
        # Использование set с merge=True для создания/обновления
        await asyncio.to_thread(user_ref.set, profile_data.model_dump(), merge=True)
        print(f"User {user_id} profile created/updated in Firestore.")
        return JSONResponse(status_code=201, content={
            "status": "success",
            "message": "Профиль пользователя успешно создан/обновлен",
            "profile": profile_data.model_dump()
        })
    except Exception as e:
        print(f"Ошибка при создании/обновлении профиля для {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при создании/обновлении профиля: {e}")

@app.get("/users/profile")
async def get_user_profile(user_id: str = Depends(get_current_user)):
    """
    Получает профиль пользователя из Firestore.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        user_ref = db.collection('users').document(user_id)
        # Получаем документ
        doc = await asyncio.to_thread(user_ref.get)
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Профиль пользователя не найден.")
        
        profile = convert_timestamps_to_strings(doc.to_dict())
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Профиль пользователя успешно получен",
            "profile": profile
        })
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Ошибка при получении профиля для {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сервера при получении профиля.")

@app.put("/users/profile")
async def update_user_profile(profile_data: UserProfileUpdate, user_id: str = Depends(get_current_user)):
    """
    Обновляет существующий профиль пользователя в Firestore.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        # Проверяем, есть ли хотя бы одно поле для обновления
        update_data = profile_data.model_dump(exclude_unset=True)

        if not update_data:
            raise HTTPException(status_code=400, detail="Нет данных для обновления.")
        
        user_ref = db.collection('users').document(user_id)
        # Обновляем документ
        await asyncio.to_thread(user_ref.update, update_data)
        print(f"User {user_id} profile updated in Firestore.")
        
        # Получаем обновленный документ для ответа
        updated_doc = await asyncio.to_thread(user_ref.get)
        updated_profile = convert_timestamps_to_strings(updated_doc.to_dict())

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Профиль пользователя успешно обновлен",
            "profile": updated_profile
        })
    except google_exceptions.NotFound:
        raise HTTPException(status_code=404, detail="Профиль пользователя не найден.")
    except Exception as e:
        print(f"Ошибка при обновлении профиля для {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сервера при обновлении профиля.")

# --- API для обучающих планов (из спецификации) ---
@app.post("/learning-plans/generate")
async def generate_learning_plan(plan_request: LearningPlanGenerateRequest, user_id: str = Depends(get_current_user)):
    if llm_model is None:
        # Используем вспомогательную функцию для стандартизированного ответа
        return generate_error_response(status_code=500, detail="LLM модель не инициализирована.")
    print(f"User {user_id} requested learning plan generation: {plan_request.model_dump()}")
    
    prompt = f"Сгенерируй детальный обучающий план по предмету '{plan_request.subject}' типа '{plan_request.planType}'. "
    if plan_request.targetGoal:
        prompt += f"Цель: {plan_request.targetGoal}. "
    if plan_request.language:
        prompt += f"На языке: {plan_request.language}. "
    if plan_request.userPreferences:
        prompt += f"Предпочтения пользователя: {plan_request.userPreferences}. "
    
    try:
        response = llm_model.generate_content(prompt)
        generated_plan_text = response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Ошибка при генерации плана с LLM: {e}")
        # Используем вспомогательную функцию для стандартизированного ответа
        return generate_error_response(status_code=500, detail=f"Ошибка при генерации плана: {e}")

    plan_id = "plan_" + os.urandom(8).hex()
    # Обычно планы сохраняются в БД перед возвратом, но по структуре спецификации
    # кажется, что тут возвращается только сгенерированный текст.
    # Добавьте сохранение в Firestore, если это требуется.
    # plan_ref = db.collection('users').document(user_id).collection('learning_plans').document(plan_id)
    # plan_ref.set(...)
    return JSONResponse(status_code=200, content={
        "status": "success",
        "message": "План успешно сгенерирован.",
        "plan": {
            "planId": plan_id,
            "subject": plan_request.subject,
            "title": f"Обучающий план по {plan_request.subject}",
            "description": generated_plan_text,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "status": "generated",
            "sections": []
        }
    })

@app.get("/learning-plans")
async def get_user_learning_plans(user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    print(f"User {user_id} requested list of learning plans.")
    plans_ref = db.collection('users').document(user_id).collection('learning_plans')
    plans = []
    for doc in plans_ref.stream():
        plan_data = doc.to_dict()
        plan_data['planId'] = doc.id
        plans.append(convert_timestamps_to_strings(plan_data))
    return JSONResponse(status_code=200, content={
        "status": "success",
        "plans": plans
    })

@app.get("/learning-plans/{planId}")
async def get_learning_plan_details(planId: str, user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    print(f"User {user_id} requested details for plan {planId}.")
    plan_ref = db.collection('users').document(user_id).collection('learning_plans').document(planId)
    plan_doc = plan_ref.get()

    if not plan_doc.exists:
        raise HTTPException(status_code=404, detail="Обучающий план не найден.")

    plan_data = plan_doc.to_dict()
    plan_data['planId'] = plan_doc.id
    plan_data = convert_timestamps_to_strings(plan_data)
    return JSONResponse(status_code=200, content={
        "status": "success",
        "plan": plan_data
    })

@app.put("/learning-plans/{planId}/progress")
async def update_learning_plan_progress(planId: str, progress_data: dict, user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    print(f"User {user_id} updated progress for plan {planId}: {progress_data}")
    plan_ref = db.collection('users').document(user_id).collection('learning_plans').document(planId)
    try:
        plan_ref.update(progress_data)
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Прогресс успешно обновлен.",
            "updatedProgress": progress_data
        })
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"Ошибка при обновлении прогресса: {e}")
# --- API для LLM-агента (чата) ---
@app.post("/agent/chat")
async def chat_with_agent(chat_message: AgentChatMessage, user_id: str = Depends(get_current_user)):
    if llm_model is None:
        raise HTTPException(status_code=500, detail="LLM модель не инициализирована.")
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    print(f"User {user_id} sent message to agent: {chat_message.model_dump()}")

    full_prompt = chat_message.message
    source_documents_info = []

    if chat_message.document_ids:
        print(f"Выполнение RAG для документов: {chat_message.document_ids}")
        try:
            context = await retrieve_relevant_context(chat_message.message, chat_message.document_ids)
            full_prompt = f"Используя следующий контекст, ответь на вопрос: '{chat_message.message}'\n\nКонтекст:\n{context}"
            
            for doc_id in chat_message.document_ids:
                doc_meta_ref = db.collection('users').document(user_id).collection('documents').document(doc_id)
                doc_meta = doc_meta_ref.get()
                if doc_meta.exists:
                    source_documents_info.append({"doc_id": doc_id, "title": doc_meta.to_dict().get("title", "Неизвестный документ")})
                else:
                    source_documents_info.append({"doc_id": doc_id, "title": "Документ не найден"})

        except HTTPException as e:
            raise e
        except Exception as e:
 # Let the global exception handler handle this
 raise e
    else:
        print("Выполнение общего запроса к LLM.")

    try:
        response = llm_model.generate_content(full_prompt)
        llm_response_text = response.candidates[0].content.parts[0].text
    except Exception as e:
 # Let the global exception handler handle this
 raise e
    conversation_id = chat_message.conversationId or "conv_" + os.urandom(8).hex()

    chat_history_ref = db.collection('users').document(user_id).collection('conversations').document(conversation_id).collection('messages')
    chat_history_ref.add({
        "sender": "user",
        "message": chat_message.message,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    chat_history_ref.add({
        "sender": "agent",
        "message": llm_response_text,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "source_documents": source_documents_info
    })

    return JSONResponse(status_code=200, content={
        "status": "success",
        "response": llm_response_text,
        "conversationId": conversation_id,
        "suggestedNextSteps": ["Продолжить", "Задать другой вопрос"],
        "sourceDocuments": source_documents_info
    })

@app.get("/agent/chat/history/{conversationId}")
async def get_chat_history(conversationId: str, user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    print(f"User {user_id} requested chat history for {conversationId}.")
    messages_ref = db.collection('users').document(user_id).collection('conversations').document(conversationId).collection('messages').order_by('timestamp')
    messages = []
    for doc in messages_ref.stream():
        msg_data = doc.to_dict()
        messages.append(convert_timestamps_to_strings(msg_data))
    return JSONResponse(status_code=200, content={
        "status": "success",
        "conversationId": conversation_id,
        "messages": messages
    })

# --- API для геймификации и прогресса (из спецификации) ---
@app.post("/gamification/complete-task")
async def complete_gamification_task(task_data: GamificationCompleteTask, user_id: str = Depends(get_current_user)):
    print(f"User {user_id} completed task: {task_data.model_dump()}")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "message": "Задача успешно записана (заглушка).",
        "userProgress": {
            "totalPoints": 150,
            "newAchievements": ["first_lesson_completed"]
        }
    })

@app.get("/gamification/achievements")
async def get_user_achievements(user_id: str = Depends(get_current_user)):
    print(f"User {user_id} requested achievements.")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "achievements": [
            {"achievementId": "first_lesson_completed", "name": "Первый урок", "description": "Завершил первый урок", "unlockedAt": "2025-07-22T11:00:00Z"}
        ]
    })

@app.get("/gamification/stats")
async def get_user_stats(user_id: str = Depends(get_current_user)):
    print(f"User {user_id} requested stats.")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "stats": {
            "totalPoints": 150,
            "completedLessonsCount": 5,
            "totalLearningTimeMinutes": 120,
            "currentLevel": 2,
            "nextLevelPointsRequired": 200
        }
    })

@app.get("/gamification/leaderboard")
async def get_leaderboard(sortBy: str = "totalPoints", limit: int = 10, offset: int = 0, user_id: str = Depends(get_current_user)):
    print(f"User {user_id} requested leaderboard (sort by: {sortBy}, limit: {limit}, offset: {offset}).")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "leaderboard": [
            {"userId": "user1", "displayName": "Лидер 1", "totalPoints": 500, "rank": 1},
            {"userId": "user2", "displayName": "Лидер 2", "totalPoints": 450, "rank": 2}
        ]
    })

# --- API для загрузки и управления файлами (с реальной логикой) ---
@app.post("/documents/upload")
async def upload_document(
    user_id: str = Depends(get_current_user),
    file: UploadFile = File(...)
):
    """
    Загружает и обрабатывает учебный документ, сохраняя его в Cloud Storage
    и индексируя для RAG.
    """
    if not storage_client or GCS_BUCKET_NAME is None:
        raise HTTPException(status_code=500, detail="Cloud Storage не инициализирован или имя бакета не установлено.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не предоставлено.")
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Модель эмбеддингов не инициализирована. Проверьте конфигурацию Vertex AI.")

    try:
        doc_id = str(uuid.uuid4())
        filename = file.filename
        file_content = await file.read()

        # 1. Сохранение файла в Cloud Storage
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_path = f"user_documents/{user_id}/{doc_id}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(file_content, content_type=file.content_type)
        gcs_public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob_path}"
        print(f"Файл '{filename}' сохранен в GCS по пути: {blob_path}")

        # 2. Извлечение текста из файла
        extracted_text = await extract_text_from_file(file_content, filename)
        if not extracted_text:
            raise HTTPException(status_code=422, detail="Не удалось извлечь текст из документа.")

        # 3. Разделение текста на фрагменты и генерация эмбеддингов
        text_chunks = await chunk_text(extracted_text)
        await store_document_chunks_in_vector_db(doc_id, text_chunks)

        # 4. Сохранение метаданных документа в Firestore
        if db is None:
            raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
        doc_metadata = {
            "id": doc_id,
            "title": filename,
            "user_id": user_id,
            "gcs_path": blob_path,
            "gcs_url": gcs_public_url,
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "processed_at": firestore.SERVER_TIMESTAMP,
            "file_type": file.content_type,
            "size": len(file_content),
        }
        doc_ref = db.collection('users').document(user_id).collection('documents').document(doc_id)
        doc_ref.set(doc_metadata)
        print(f"Метаданные документа {doc_id} сохранены в Firestore.")

        return JSONResponse(status_code=201, content={
            "status": "success",
            "message": "Документ успешно загружен и обработан.",
            "doc_id": doc_id,
            "title": filename,
            "gcs_url": gcs_public_url
        })

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
 # Let the global exception handler handle this
 raise e
@app.get("/files/download-url/{fileId}")
async def get_download_url(fileId: str, user_id: str = Depends(get_current_user)):
    if db is None or storage_client is None or GCS_BUCKET_NAME is None:
        raise HTTPException(status_code=500, detail="Firestore DB, Cloud Storage или имя бакета не инициализированы.")
    doc_ref = db.collection('users').document(user_id).collection('documents').document(fileId)
    doc_data = doc_ref.get()
    if not doc_data.exists:
        raise HTTPException(status_code=404, detail="Файл не найден.")
    
    gcs_path = doc_data.to_dict().get("gcs_path")
    if not gcs_path:
        raise HTTPException(status_code=500, detail="Путь к файлу в GCS не найден.")

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        signed_url = blob.generate_signed_url(expiration=3600, method='GET')
        return JSONResponse(status_code=200, content={
            "status": "success",
            "downloadUrl": signed_url
        })
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"Ошибка при генерации URL для скачивания: {e}")
@app.get("/files")
async def get_user_files(user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    print(f"User {user_id} requested list of files.")
    files_ref = db.collection('users').document(user_id).collection('documents')
    files = []
    for doc in files_ref.stream():
        file_data = doc.to_dict()
        file_data['fileId'] = doc.id
        files.append(convert_timestamps_to_strings(file_data))
    return JSONResponse(status_code=200, content={
        "status": "success",
        "files": files
    })

# --- API для монетизации (из спецификации) ---
@app.get("/billing/free-limit-status")
async def get_free_limit_status(user_id: str = Depends(get_current_user)):
    print(f"User {user_id} requested free limit status.")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "freeLimitRemaining": 50,
        "limitType": "llm_queries",
        "isPremiumUser": False,
        "nextResetDate": "2025-08-01T00:00:00Z"
    })

@app.post("/billing/purchase-webhook")
async def handle_purchase_webhook(webhook_data: BillingPurchaseWebhook):
    print(f"Received purchase webhook: {webhook_data.model_dump()}")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "message": "Уведомление о покупке успешно обработано (заглушка)."
    })

@app.get("/billing/subscription-status")
async def get_subscription_status(user_id: str = Depends(get_current_user)):
    print(f"User {user_id} requested subscription status.")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "isSubscribed": True,
        "subscriptionType": "premium_monthly",
        "expiresAt": "2025-08-22T00:00:00Z",
        "autoRenewing": True
    })

# --- Добавлен блок для запуска Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    