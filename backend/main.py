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
# --- Правильные, публичные импорты для Firestore ---
from google.cloud.firestore import Timestamp
from google.cloud.firestore_v1.collection import CollectionReference

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

# --- Имитация векторной базы данных в памяти (для RAG) ---
# Эти переменные должны быть определены на глобальном уровне
MOCKED_VECTOR_DB: Dict[str, Dict[str, Any]] = {}
MOCKED_DOC_CHUNKS_MAP: Dict[str, List[str]] = {}

# --- Глобальные переменные для Firebase и GCP ---
# Получаем ID проекта и имя бакета из переменных окружения
# Если они не установлены, это указывает на проблему с развертыванием
# и приложение не сможет функционировать корректно.
# Убедитесь, что эти переменные окружения установлены в Cloud Run!
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME") # Общий бакет для документов и т.д.
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

db = None # Firestore client
try:
    if not firebase_admin._apps: # Проверяем, не инициализировано ли уже приложение
        firebase_admin.initialize_app(credentials.ApplicationDefault())
    db = firestore.client()
    logging.info("Firebase Admin SDK успешно инициализирован.")
except Exception as e:
    logging.error(f"Ошибка инициализации Firebase Admin SDK: {e}", exc_info=True)
    # Это критическая ошибка, без которой приложение не может работать.
    # Лучше "упасть" сразу с понятным сообщением.
    raise RuntimeError(f"Не удалось инициализировать Firebase Admin SDK: {e}") from e

# --- Инициализация Vertex AI ---
llm_model = None
embedding_model = None
try:
    if GCP_PROJECT_ID and GCP_LOCATION:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        llm_model = GenerativeModel("gemini-1.5-flash-preview-0514")
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        logging.info(f"Vertex AI initialized for project {GCP_PROJECT_ID} in {GCP_LOCATION}")
    else:
        logging.warning("Пропуск инициализации Vertex AI: GCP_PROJECT_ID или GCP_LOCATION не установлены.")
except Exception as e:
    logging.error(f"Предупреждение: Не удалось инициализировать Vertex AI: {e}. Функции LLM будут недоступны.", exc_info=True)

# --- Инициализация клиента Cloud Storage ---
storage_client: Optional[storage.Client] = None # Используйте type hint для яскости

try:
    if GCP_PROJECT_ID:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        logging.info("Cloud Storage Client успешно инициализирован.") # Используйте logging
    else:
        logging.warning("Пропуск инициализации Cloud Storage Client: GCP_PROJECT_ID не установлен.") # Используйте logging
except Exception as e:
    logging.error(f"Предупреждение: Не удалось инициализировать Cloud Storage Client: {e}. Функции хранилища будут недоступны.", exc_info=True) # Логируйте ошибку детально

CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS")

if CORS_ALLOWED_ORIGINS:
    origins = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",")]
else:
    # Фоллбэк на жестко закодированный список для локальной разработки,
    # если переменная окружения не установлена.
    # В продакшене настоятельно рекомендуется использовать переменную окружения.
    logging.warning("Предупреждение: Переменная окружения CORS_ALLOWED_ORIGINS не установлена. Используется список по умолчанию для разработки.")
    origins = [
        "https://ailbee.web.app",
        "https://ailbee.firebaseapp.com",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:8000",
    ]

logging.info(f"CORS настроен для следующих источников: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Вспомогательная функция для генерации стандартизированных ответов об ошибках ---\
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
    # Возвращаем стандартизированный ответ об ошибке
    return generate_error_response(status_code=500, detail="Произошла внутренняя ошибка сервера.")

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

@app.on_event("startup")
async def startup_event():
    logging.info("Запуск приложения: Выполнение инициализации коллекций...")
    await initialize_collections()
    logging.info("Запуск приложения: Инициализация коллекций завершена.")

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
        # === Вот здесь происходит верификация токена и получение UID ===
        decoded_token = auth.verify_id_token(token)
        request.state.user_id = decoded_token['uid'] # Сохраняем UID в состоянии запроса (опционально, но удобно)
        return decoded_token['uid'] # Возвращаем UID пользователя
    except Exception as e:
        # Лучше логировать ошибку, чтобы видеть детали в логах сервера
        logging.error(f"Ошибка верификации токена: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Недействительный токен: {e}")

class SummarizeTextRequest(BaseModel):
    text: str

# --- Вспомогательная функция для преобразования Firestore Timestamp в строки ---
def convert_timestamps_to_strings(data: Any) -> Any:
    """
    Рекурсивно преобразует объекты Firestore Timestamp и datetime в строки ISO 8601
    внутри словарей и списков.
    """
    # Исправленная проверка типа. Используем импортированный Timestamp.
    if isinstance(data, (Timestamp, datetime.datetime)):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: convert_timestamps_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps_to_strings(elem) for elem in data]
    else:
        return data

# --- API для загрузки пользовательских данных ---
class UploadResponse(BaseModel):
    status: str
    message: str
    file_url: str

@app.post("/upload_user_data", response_model=UploadResponse)
async def upload_user_data(user_data_file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """
    Загружает файл, предоставленный пользователем, в бакет \'ailbee-user-data\'.
    Файл сохраняется в директорию, специфичную для пользователя.
    
    Args:
        user_data_file: Файл для загрузки.
    """
    logging.info(f"Получен запрос на загрузку пользовательских данных от {user_id}")

    try:
        if storage_client is None:
            raise HTTPException(status_code=500, detail="Cloud Storage не инициализирован.")
        bucket_name = GCS_USER_DATA_BUCKET or "ailbee-user-data"
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось получить доступ к бакету: {str(e)}")

    if not user_data_file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не предоставлено.")
    
    file_extension = user_data_file.filename.split('.')[-1]
    file_path = f"users/{user_id}/{uuid.uuid4()}.{file_extension}"
    blob = bucket.blob(file_path)

    try:
        file_bytes = await user_data_file.read()
        await asyncio.to_thread(blob.upload_from_string, file_bytes, content_type=user_data_file.content_type)
        file_url = blob.public_url
        logging.info(f"Файл '{user_data_file.filename}' успешно загружен в бакет {bucket_name} по пути {file_path}")
        
        if db is None:
            raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
        
        user_files_ref = db.collection('users').document(user_id).collection('files')
        await asyncio.to_thread(user_files_ref.add, {
            "file_name": user_data_file.filename,
            "file_url": file_url,
            "content_type": user_data_file.content_type,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Метаданные файла сохранены в Firestore для пользователя {user_id}")

        return {"status": "success", "message": "Файл успешно загружен.", "file_url": file_url}

    except Exception as e:
        logging.error(f"Ошибка при загрузке пользовательских данных: {e}", exc_info=True)
        # Используем HTTPException для обработки глобальным обработчиком
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
    
    Args:
    profile_picture: Файл изображения профиля.
    """
    logging.info(f"Получен запрос на загрузку изображения профиля от пользователя: {user_id}")

    try:
        if storage_client is None:
            raise HTTPException(status_code=500, detail="Cloud Storage не инициализирован.")
        bucket_name = GCS_USER_DATA_BUCKET or "ailbee-user-data"
        bucket = storage_client.bucket(bucket_name)
        
        if not profile_picture.filename:
            raise HTTPException(status_code=400, detail="Имя файла не предоставлено.")
        
        file_extension = profile_picture.filename.split('.')[-1]
        file_path = f"profiles/{user_id}.{file_extension}"
        blob = bucket.blob(file_path)

        file_bytes = await profile_picture.read()
        await asyncio.to_thread(blob.upload_from_string, file_bytes, content_type=profile_picture.content_type)
        
        file_url = blob.public_url
        logging.info(f"Изображение профиля успешно загружено по пути: {file_path}")

        if db is None:
            raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
        
        user_doc_ref = db.collection('users').document(user_id)
        await asyncio.to_thread(user_doc_ref.set, {"profile_picture_url": file_url}, merge=True)
        logging.info(f"Ссылка на изображение профиля успешно сохранена в Firestore для пользователя {user_id}")

        return {"status": "success", "message": "Изображение профиля успешно загружено.", "file_url": file_url}

    except HTTPException as http_exc:
        raise http_exc # Перебрасываем HTTPException
    except Exception as e:
        logging.error(f"Ошибка при загрузке изображения профиля: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при загрузке файла.")

# --- API для получения информации об экспортах пользователя ---
@app.get("/users/me/export_logs")
async def get_user_export_logs(user_id: str = Depends(get_current_user)):
    """
    Получает логи экспорта данных для текущего аутентифицированного пользователя.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        # Ищем документы в коллекции export_logs, связанные с этим пользователем
        # Используем поле 'userId', которое мы добавляем в Cloud Function processExportedUserData
        export_logs_ref = db.collection('export_logs').where('userId', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        docs = await asyncio.to_thread(export_logs_ref.stream)
        
        export_list = []
        # Используем вспомогательную функцию convert_timestamps_to_strings
        for doc in docs:
            export_list.append(convert_timestamps_to_strings(doc.to_dict() or {}))

        logging.info(f"Получены логи экспорта для пользователя {user_id}: {len(export_list)} записей.")

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Логи экспорта успешно получены",
            "export_logs": export_list
        })
    except Exception as e:
        logging.error(f"Ошибка при получении логов экспорта для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при получении логов экспорта.")

# --- Инициализация коллекций с начальными данными ---
async def initialize_collections():
    """
    Инициализирует коллекции subjects и educationLevels, если они пусты.
    Эта функция выполняется при запуске приложения.
    """
    if db is None:
        logging.warning("Пропуск инициализации коллекций: Firestore DB не инициализирован.")
        return
    # Используем Type Hint для ясности
    subjects_ref: CollectionReference = db.collection('subjects')
    education_levels_ref: CollectionReference = db.collection('educationLevels')

    try:
        # Проверяем, пуста ли коллекция subjects
        subjects_docs = await asyncio.to_thread(subjects_ref.limit(1).get)
        if not subjects_docs: # Проверяем, пустой ли список документов
            logging.info("Инициализация коллекции 'subjects'...")
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
            logging.info("Коллекция 'subjects' успешно инициализирована.")
        else:
            logging.info("Коллекция 'subjects' уже содержит данные. Пропуск инициализации.")

        # Проверяем, пуста ли коллекция educationLevels
        education_levels_docs = await asyncio.to_thread(education_levels_ref.limit(1).get)
        if not education_levels_docs: # Проверяем, пустой ли список документов
            logging.info("Инициализация коллекции 'educationLevels'...")
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
            logging.info("Коллекция 'educationLevels' успешно инициализирована.")
        else:
            logging.info("Коллекция 'educationLevels' уже содержит данные. Пропуск инициализации.")

    except Exception as e:
        logging.error(f"Ошибка при инициализации коллекций: {e}", exc_info=True)

@app.post("/upload_book")
async def upload_book(book_file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    try:
        if storage_client is None:
             raise HTTPException(status_code=500, detail="Cloud Storage не инициализирован.")
        if db is None:
             raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")

        # 1. Выбираем бакет для книг
        bucket_name = GCS_BOOKS_BUCKET or "ailbee-books"
        bucket = storage_client.bucket(bucket_name)

        if not book_file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не предоставлено.")

        # 2. Создаем уникальное имя файла
        file_extension = book_file.filename.split('.')[-1]
        file_path = f"books/{user_id}/{uuid.uuid4()}.{file_extension}"
        blob = bucket.blob(file_path)

        # 3. Загружаем файл в бакет
        file_bytes = await book_file.read()
        await asyncio.to_thread(blob.upload_from_string, file_bytes)

        # 4. Получаем URL загруженного файла
        file_url = blob.public_url
        logging.info(f"Файл успешно загружен в бакет {bucket_name}.")

        # 5. Сохраняем ссылку на файл в коллекции Firestore
        books_collection_ref = db.collection('users').document(user_id).collection('books')
        update_time, book_doc_ref = await asyncio.to_thread(books_collection_ref.add, {
            "title": book_file.filename,
            "storage_url": file_url,
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "status": "pending_processing"
        })

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Книга успешно загружена и добавлена в Firestore.",
            "book_id": book_doc_ref.id
        })

    except Exception as e:
        logging.error(f"Ошибка загрузки книги: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при загрузке книги.")

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
            # Для других типов файлов пытаемся декодировать как текст
            return file_content.decode('utf-8', errors='ignore')
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из файла {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail="Не удалось извлечь текст из файла.")

async def generate_embedding_for_text(text: str) -> List[float]:
    """
    Генерирует векторные эмбеддинги для заданного текста с помощью Vertex AI.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Модель эмбеддингов не инициализирована.")
    try:
        # Исправлено: .embed() возвращает список эмбеддингов, нужно взять первый элемент.
        embeddings = embedding_model.embed([text])
        return embeddings[0].values
    except Exception as e:
        logging.error(f"Ошибка при генерации эмбеддингов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при генерации эмбеддингов.")

async def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Разделяет текст на перекрывающиеся фрагменты.
    """
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

async def store_document_chunks_in_vector_db(doc_id: str, text_chunks: List[str]):
    chunk_ids = []
    for i, chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        embedding = await generate_embedding_for_text(chunk)
        MOCKED_VECTOR_DB[chunk_id] = {"text": chunk, "embedding": embedding}
        chunk_ids.append(chunk_id)
    MOCKED_DOC_CHUNKS_MAP[doc_id] = chunk_ids
    logging.info(f"Документ {doc_id} и его фрагменты сохранены в имитированной векторной БД.")

async def retrieve_relevant_context(query: str, document_ids: List[str]) -> str:
    """
    Извлекает наиболее релевантный контекст из имитированной векторной БД,
    используя косинусное сходство.
    """
    top_k = 5
    logging.info(f"Извлечение {top_k} наиболее релевантных фрагментов для запроса: '{query}' из документов: {document_ids}")

    if not document_ids:
        return "Документы для поиска не указаны."

    try:
        query_embedding = np.array(await generate_embedding_for_text(query))
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Ошибка при генерации эмбеддинга для запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось сгенерировать эмбеддинг для запроса.")

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

    for chunk in candidate_chunks:
        chunk['similarity'] = np.dot(query_embedding, chunk['embedding'])

    sorted_chunks = sorted(candidate_chunks, key=lambda x: x['similarity'], reverse=True)
    top_chunks = sorted_chunks[:top_k]

    if not top_chunks:
        return "Не удалось найти релевантный контекст в загруженных документах."

    context_texts = [chunk['text'] for chunk in top_chunks]
    full_context = "\n\n---\n\n".join(context_texts)
    MAX_CONTEXT_LENGTH = 7000
    return full_context[:MAX_CONTEXT_LENGTH]

# --- API Endpoints ---

@app.get("/")
async def root():
    logging.info('Received request for /')
    return {"message": "Welcome to FastAPI Backend! Service is running."}

@app.get("/test")
async def test_endpoint():
    logging.info('Received request for /test')
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB failed to initialize!")
    return {"message": "Hello from Cloud Run! Firestore DB is connected."}

@app.post("/save_data")
async def save_generic_data(payload: FrontendSaveDataPayload, user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB not initialized.")
    data_to_save = payload.data
    data_to_save["timestamp"] = firestore.SERVER_TIMESTAMP

    try:
        doc_ref = db.collection('users').document(user_id).collection('generic_data').document('my_data')
        await asyncio.to_thread(doc_ref.set, data_to_save)
        logging.info(f"Generic data saved for user {user_id} in Firestore.")
        return {"status": "success", "message": "Данные успешно сохранены!", "saved_data": data_to_save}
    except Exception as e:
        logging.error(f"Error saving data to Firestore for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при сохранении данных.")

@app.get("/get_data")
async def get_generic_data(user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB not initialized.")
    try:
        doc_ref = db.collection('users').document(user_id).collection('generic_data').document('my_data')
        doc = await asyncio.to_thread(doc_ref.get)

        if not doc.exists:
            logging.info(f"No generic data found for user {user_id}.")
            return {"status": "success", "message": "Данные не найдены.", "data": None}

        data = doc.to_dict()
        data = convert_timestamps_to_strings(data)
        logging.info(f"Generic data retrieved for user {user_id} from Firestore.")
        return {"status": "success", "message": "Данные успешно получены!", "data": data}
    except Exception as e:
        logging.error(f"Error retrieving data from Firestore for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при получении данных.")

# --- API для обработки событий от Eventarc ---
@app.post("/eventarc-handler")
async def handle_eventarc_event(request: Request):
    """
    Обрабатывает входящие события CloudEvents от Eventarc.
    """
    event_type = request.headers.get("ce-type")
    subject = request.headers.get("ce-subject")
    
    logging.info(f"Получено событие Eventarc типа '{event_type}' для субъекта '{subject}'")

    try:
        body = await request.json()
    except Exception as e:
        logging.error(f"Ошибка парсинга тела запроса Eventarc: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Некорректный формат тела запроса (ожидается JSON).")

    if event_type == "google.cloud.storage.object.v1.finalized":
        bucket = body.get("bucket")
        file_name = body.get("name")
        logging.info(f"Новый файл '{file_name}' был загружен в бакет '{bucket}'.")

    return Response(status_code=204)

@app.post("/library/books")
async def create_library_book(
    book_data: LibraryBookCreate,
    user_id: str = Depends(get_current_user)):
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
        logging.info(f"Книга '{book_data.title}' (ID: {book_id}) добавлена в библиотеку.")

        response_book_data = book_dict.copy()
        response_book_data["bookId"] = book_id

        return JSONResponse(status_code=201, content={
            "status": "success",
            "message": "Книга успешно добавлена в библиотеку.",
            "bookId": book_id,
            "book": response_book_data
        })
    except Exception as e:
        logging.error(f"Ошибка при добавлении книги: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при добавлении книги в библиотеку.")

@app.get("/library/books")
async def get_library_books(
    user_id: str = Depends(get_current_user),
    education_level: Optional[str] = Query(None, description="Фильтр по уровню образования"),
    subject: Optional[str] = Query(None, description="Фильтр по предмету")):
    """
    Получает список книг из центральной библиотеки с возможностью фильтрации.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        logging.info(f"Пользователь {user_id} запросил список книг из библиотеки. Фильтры: уровень={education_level}, предмет={subject}")

        query: CollectionReference = db.collection('libraryBooks')

        if education_level:
            query = query.where('education_level', '==', education_level)
        if subject:
            query = query.where('subject', '==', subject)

        books = []
        for doc in await asyncio.to_thread(query.stream):
            book_data = doc.to_dict()
            if book_data:
                book_data['bookId'] = doc.id
                books.append(convert_timestamps_to_strings(book_data))

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Список книг успешно получен.",
            "books": books
        })
    except google_exceptions.FailedPrecondition as e:
        error_detail = f"Query failed. A required Firestore index is likely missing. Check backend logs for details. Original error: {e}"
        logging.error(error_detail, exc_info=True)
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        logging.error(f"Ошибка при получении списка книг: {e}", exc_info=True)
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
        response = await asyncio.to_thread(llm_model.generate_content, prompt)
        summarized_text = response.candidates[0].content.parts[0].text
        
        logging.info(f"Текст успешно суммирован для пользователя {user_id}.")
        return JSONResponse(status_code=200, content={
            "status": "success",
            "summary": summarized_text
        })
    except Exception as e:
        logging.error(f"Ошибка при суммировании текста: {e}", exc_info=True)
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
        await asyncio.to_thread(user_ref.set, profile_data.model_dump(), merge=True)
        logging.info(f"User {user_id} profile created/updated in Firestore.")
        return JSONResponse(status_code=201, content={
            "status": "success",
            "message": "Профиль пользователя успешно создан/обновлен",
            "profile": profile_data.model_dump()
        })
    except Exception as e:
        logging.error(f"Ошибка при создании/обновлении профиля для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сервера при создании/обновлении профиля.")

@app.get("/users/profile")
async def get_user_profile(user_id: str = Depends(get_current_user)):
    """
    Получает профиль пользователя из Firestore.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        user_ref = db.collection('users').document(user_id)
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
        logging.error(f"Ошибка при получении профиля для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сервера при получении профиля.")

@app.put("/users/profile")
async def update_user_profile(profile_data: UserProfileUpdate, user_id: str = Depends(get_current_user)):
    """
    Обновляет существующий профиль пользователя в Firestore.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    try:
        update_data = profile_data.model_dump(exclude_unset=True)

        if not update_data:
            raise HTTPException(status_code=400, detail="Нет данных для обновления.")
        
        user_ref = db.collection('users').document(user_id)
        await asyncio.to_thread(user_ref.update, update_data)
        logging.info(f"User {user_id} profile updated in Firestore.")
        
        updated_doc = await asyncio.to_thread(user_ref.get)
        updated_profile = convert_timestamps_to_strings(updated_doc.to_dict())

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Профиль пользователя успешно обновлен",
            "profile": updated_profile # type: ignore
        })
    except google_exceptions.NotFound:
        raise HTTPException(status_code=404, detail="Профиль пользователя не найден.")
    except Exception as e:
        logging.error(f"Ошибка при обновлении профиля для {user_id}: {e}", exc_info=True) # Изменено на logging.error с exc_info
        raise HTTPException(status_code=500, detail="Ошибка сервера при обновлении профиля.")

@app.post("/learning-plans/generate")
async def generate_learning_plan(plan_request: LearningPlanGenerateRequest, user_id: str = Depends(get_current_user)):
    if llm_model is None:
        raise HTTPException(status_code=500, detail="LLM модель не инициализирована.")
    logging.info(f"User {user_id} requested learning plan generation: {plan_request.model_dump()}")
    
    prompt = f"Сгенерируй детальный обучающий план по предмету '{plan_request.subject}' типа '{plan_request.planType}'. "
    if plan_request.targetGoal:
        prompt += f"Цель: {plan_request.targetGoal}. "
    if plan_request.language:
        prompt += f"На языке: {plan_request.language}. "
    if plan_request.userPreferences:
        prompt += f"Предпочтения пользователя: {plan_request.userPreferences}. "
    
    try:
        response = await asyncio.to_thread(llm_model.generate_content, prompt) # Используем asyncio.to_thread
        generated_plan_text = response.candidates[0].content.parts[0].text # type: ignore
    except Exception as e:
        logging.error(f"Ошибка при генерации плана с LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации плана.") # Более общее сообщение для пользователя

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
            "createdAt": firestore.SERVER_TIMESTAMP, # Consider converting to string here
            "status": "generated",
            "sections": [] # This might need actual sections generated by LLM or further processing
        }
    })

@app.get("/learning-plans")
async def get_user_learning_plans(user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    logging.info(f"User {user_id} requested list of learning plans.")
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
    # Перебрасываем HTTPException при ошибках Firebase/Firestore
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    logging.info(f"User {user_id} requested details for plan {planId}.")
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
    logging.info(f"User {user_id} updated progress for plan {planId}: {progress_data}")
    plan_ref = db.collection('users').document(user_id).collection('learning_plans').document(planId)
    try:
        # Проверяем, существует ли план перед обновлением (опционально, но хорошая практика)
        plan_doc = await asyncio.to_thread(plan_ref.get)
        if not plan_doc.exists:
             raise HTTPException(status_code=404, detail="Обучающий план не найден.")

        await asyncio.to_thread(plan_ref.update, progress_data)
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "Прогресс успешно обновлен.",
            "updatedProgress": progress_data
        })
    except google_exceptions.NotFound: # Ловим специфическую ошибку Firestore
        raise HTTPException(status_code=404, detail="Обучающий план не найден для обновления.")
    except Exception as e:
        logging.error(f"Ошибка при обновлении прогресса для плана {planId} пользователя {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обновлении прогресса.")
@app.post("/agent/chat")
async def chat_with_agent(chat_message: AgentChatMessage, user_id: str = Depends(get_current_user)):
    if llm_model is None:
        raise HTTPException(status_code=500, detail="LLM модель не инициализирована.")
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    logging.info(f"User {user_id} sent message to agent: {chat_message.model_dump()}")

    full_prompt = chat_message.message
    source_documents_info = []

    if chat_message.document_ids:
        logging.info(f"Выполнение RAG для документов: {chat_message.document_ids}")
        try:
            context = await retrieve_relevant_context(chat_message.message, chat_message.document_ids)
            full_prompt = f"Используя следующий контекст, ответь на вопрос: '{chat_message.message}'\n\nКонтекст:\n{context}"
            
            for doc_id in chat_message.document_ids:
                doc_meta_ref = db.collection('users').document(user_id).collection('documents').document(doc_id)
                doc_meta = await asyncio.to_thread(doc_meta_ref.get) # Используем asyncio.to_thread
                if doc_meta.exists:
                    source_documents_info.append({"doc_id": doc_id, "title": doc_meta.to_dict().get("title", "Неизвестный документ")})
                else:
                    source_documents_info.append({"doc_id": doc_id, "title": "Документ не найден"}) # Можно логировать предупреждение, если документ не найден

        except HTTPException as e:
            raise e
        except Exception as e:
            # Let the global exception handler handle this
            logging.error(f"Ошибка при выполнении RAG для пользователя {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Произошла ошибка при поиске контекста для ответа.")
    else:
        logging.info("Выполнение общего запроса к LLM.")

    try:
        response = await asyncio.to_thread(llm_model.generate_content, full_prompt) # Используем asyncio.to_thread
        llm_response_text = response.candidates[0].content.parts[0].text
    except Exception as e:
        logging.error(f"Ошибка при генерации ответа LLM для пользователя {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при генерации ответа LLM.")

    conversation_id = chat_message.conversationId or "conv_" + os.urandom(8).hex()

    # Логика сохранения истории чата
    try:
        chat_history_ref = db.collection('users').document(user_id).collection('conversations').document(conversation_id).collection('messages')
        await asyncio.to_thread(chat_history_ref.add, {
            "sender": "user",
            "message": chat_message.message,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        await asyncio.to_thread(chat_history_ref.add, {
            "sender": "agent",
            "message": llm_response_text,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "source_documents": source_documents_info
        })
    except Exception as e:
        logging.error(f"Ошибка при сохранении истории чата для пользователя {user_id} (конверсия {conversation_id}): {e}", exc_info=True)
        # В этом случае, возможно, не нужно бросать HTTPException, чтобы пользователь получил ответ,
        # но нужно знать, что сохранение истории не удалось.

    return JSONResponse(status_code=200, content={
        "status": "success",
        "response": llm_response_text,
        "conversationId": conversation_id,
        "suggestedNextSteps": ["Продолжить", "Задать другой вопрос"], # TODO: Generate dynamically?
        "sourceDocuments": source_documents_info # Передача информации об источниках
    })

@app.get("/agent/chat/history/{conversationId}")
async def get_chat_history(conversationId: str, user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    logging.info(f"User {user_id} requested chat history for {conversationId}.")
    messages_ref = db.collection('users').document(user_id).collection('conversations').document(conversationId).collection('messages').order_by('timestamp')
    messages = []
    # Используем asyncio.to_thread для блокирующей операции stream()
    for doc in await asyncio.to_thread(messages_ref.stream):
        msg_data = doc.to_dict()
        messages.append(convert_timestamps_to_strings(msg_data))
    return JSONResponse(status_code=200, content={
        "status": "success",
        "conversationId": conversationId, # Исправлено на conversationId вместо conversation_id
        "messages": messages
    })

# --- API для геймификации и прогресса (из спецификации) ---
@app.post("/gamification/complete-task")
async def complete_gamification_task(task_data: GamificationCompleteTask, user_id: str = Depends(get_current_user)):
    logging.info(f"User {user_id} completed task: {task_data.model_dump()}")
    # TODO: Add real logic for updating user progress, achievements, and potentially triggering Cloud Functions
    return JSONResponse(status_code=200, content={ # This endpoint is likely a stub and needs real logic
        "status": "success",
        "message": "Задача успешно записана (заглушка).",
        "userProgress": {
            "totalPoints": 150,
            "newAchievements": ["first_lesson_completed"]
        }
    })

@app.get("/gamification/achievements")
async def get_user_achievements(user_id: str = Depends(get_current_user)):
    logging.info(f"User {user_id} requested achievements.")
    # TODO: Retrieve real achievements data from Firestore or other source
    return JSONResponse(status_code=200, content={
        "status": "success",
        "achievements": [
            # Это заглушка, замените на реальные данные
            {"achievementId": "first_lesson_completed", "name": "Первый урок", "description": "Завершил первый урок", "unlockedAt": "2025-07-22T11:00:00Z"}
        ]
    })

@app.get("/gamification/stats")
async def get_user_stats(user_id: str = Depends(get_current_user)):
    logging.info(f"User {user_id} requested stats.")
    # TODO: Retrieve real stats data from Firestore or other source
    return JSONResponse(status_code=200, content={
        "status": "success",
        "stats": {
            # Это заглушка, замените на реальные данные
            "totalPoints": 150,
            "completedLessonsCount": 5,
            "totalLearningTimeMinutes": 120,
            "currentLevel": 2,
            "nextLevelPointsRequired": 200
        }
    })

@app.get("/gamification/leaderboard")
async def get_leaderboard(sortBy: str = "totalPoints", limit: int = 10, offset: int = 0, user_id: str = Depends(get_current_user)):
    logging.info(f"User {user_id} requested leaderboard (sort by: {sortBy}, limit: {limit}, offset: {offset}).")
    # TODO: Implement real leaderboard logic (query Firestore, sort, limit, offset)
    return JSONResponse(status_code=200, content={
        "status": "success",
        "leaderboard": [
            # Это заглушка, замените на реальные данные из Firestore
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
    # Перебрасываем HTTPException при ошибках
    """
    if storage_client is None or GCS_BUCKET_NAME is None: # Исправлено на is None
        raise HTTPException(status_code=500, detail="Cloud Storage не инициализирован или имя бакета не установлено.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не предоставлено.")
    if embedding_model is None: # Исправлено на is None
        raise HTTPException(status_code=500, detail="Модель эмбеддингов не инициализирована. Проверьте конфигурацию Vertex AI.")
    if db is None: # Добавлена проверка на db
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")

    try:
        doc_id = str(uuid.uuid4())
        filename = file.filename
        file_content = await file.read()

        # 1. Сохранение файла в Cloud Storage
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_path = f"user_documents/{user_id}/{doc_id}/{filename}"
        blob = bucket.blob(blob_path)
        # Используем asyncio.to_thread для блокирующей операции upload_from_string
        await asyncio.to_thread(blob.upload_from_string, file_content, content_type=file.content_type)
        gcs_public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob_path}" # TODO: Consider using signed URLs for private buckets
        logging.info(f"Файл '{filename}' сохранен в GCS по пути: {blob_path}")

        # 2. Извлечение текста из файла
        extracted_text = await extract_text_from_file(file_content, filename)
        if not extracted_text:
            # extract_text_from_file уже бросает HTTPException с 422
            # Если этот HTTPException не был брошен, но текст пустой, бросаем свой HTTPException
            raise HTTPException(status_code=422, detail="Не удалось извлечь текст из документа.")

        # 3. Разделение текста на фрагменты и генерация эмбеддингов
        text_chunks = await chunk_text(extracted_text)
        # Обработка ошибок generate_embedding_for_text и store_document_chunks_in_vector_db происходит внутри этих функций
        await store_document_chunks_in_vector_db(doc_id, text_chunks)

        # 4. Сохранение метаданных документа в Firestore
        doc_metadata = {
            "id": doc_id,
            "title": filename,
            "user_id": user_id,
            "gcs_path": blob_path,
            "gcs_url": gcs_public_url,
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "processed_at": firestore.SERVER_TIMESTAMP, # Указывает, что первичная обработка (извлечение текста, эмбеддинги) завершена
            "file_type": file.content_type,
            "size": len(file_content),
        }
        doc_ref = db.collection('users').document(user_id).collection('documents').document(doc_id)
        # Используем asyncio.to_thread для блокирующей операции set
        await asyncio.to_thread(doc_ref.set, doc_metadata)
        logging.info(f"Метаданные документа {doc_id} сохранены в Firestore.")

        return JSONResponse(status_code=201, content={
            "status": "success",
            "message": "Документ успешно загружен и обработан.",
            "doc_id": doc_id,
            "title": filename,
            "gcs_url": gcs_public_url
        })

    except HTTPException as http_exc:
        # Перебрасываем HTTPException, чтобы он был пойман глобальным обработчиком
        raise http_exc
    except Exception as e:
        # Для всех остальных неожиданных ошибок
        logging.error(f"Неожиданная ошибка при загрузке документа для пользователя {user_id}: {e}", exc_info=True)
        # Глобальный обработчик Exception поймает это и вернет 500
        raise e # Позволяем исключению подняться

@app.get("/files")
async def get_user_files(user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore DB не инициализирован.")
    logging.info(f"User {user_id} requested list of files.")
    files_ref = db.collection('users').document(user_id).collection('documents')
    files = []
    # Используем asyncio.to_thread для блокирующей операции stream()
    for doc in await asyncio.to_thread(files_ref.stream):
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
    logging.info(f"User {user_id} requested free limit status.")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "freeLimitRemaining": 50,
        "limitType": "llm_queries",
        "isPremiumUser": False,
        "nextResetDate": "2025-08-01T00:00:00Z"
    }) # This endpoint is likely a stub and needs real logic

@app.post("/billing/purchase-webhook")
async def handle_purchase_webhook(webhook_data: BillingPurchaseWebhook):
    print(f"Received purchase webhook: {webhook_data.model_dump()}")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "message": "Уведомление о покупке успешно обработано (заглушка)."
    })

@app.get("/billing/subscription-status")
async def get_subscription_status(user_id: str = Depends(get_current_user)):
    logging.info(f"User {user_id} requested subscription status.")
    return JSONResponse(status_code=200, content={
        "status": "success",
        "isSubscribed": True,
        "subscriptionType": "premium_monthly",
        "expiresAt": "2025-08-22T00:00:00Z",
        "autoRenewing": True
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
