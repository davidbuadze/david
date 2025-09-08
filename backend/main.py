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
import requests

import numpy as np
# Firebase Admin SDK
import firebase_admin
import logging
from firebase_admin import credentials, auth, firestore
from fastapi.middleware.cors import CORSMiddleware
from google.api_core import exceptions as google_exceptions

# --- Стабильный импорт CollectionReference -- -
from google.cloud.firestore_v1.collection import CollectionReference

# Google Cloud Libraries for LLM and Storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel
from google.cloud import storage

# Libraries for document parsing
from pypdf import PdfReader
from docx import Document

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- Имитация векторной базы данных в памяти (для RAG) ---
MOCKED_VECTOR_DB: Dict[str, Dict[str, Any]] = {}
MOCKED_DOC_CHUNKS_MAP: Dict[str, List[str]] = {}

# --- Глобальные переменные для Firebase и GCP ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
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

# --- Инициализация сервисов ---
app = FastAPI(
    title="Обучающее Приложение Backend API",
    description="API для управления пользователями, обучающими планами, LLM-агентом и RAG.",
    version="0.1.0",
)

db = None
storage_client: Optional[storage.Client] = None
llm_model = None
embedding_model = None

try:
    # Firebase
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.ApplicationDefault())
    db = firestore.client()
    logging.info("Firebase Admin SDK успешно инициализирован.")

    # Cloud Storage
    if GCP_PROJECT_ID:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        logging.info("Cloud Storage Client успешно инициализирован.")
    else:
        logging.warning("Пропуск инициализации Cloud Storage: GCP_PROJECT_ID не установлен.")

    # Vertex AI
    if GCP_PROJECT_ID and GCP_LOCATION:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        llm_model = GenerativeModel("gemini-1.5-flash-preview-0514")
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        logging.info(f"Vertex AI initialized for project {GCP_PROJECT_ID} in {GCP_LOCATION}")
    else:
        logging.warning("Пропуск инициализации Vertex AI: GCP_PROJECT_ID или GCP_LOCATION не установлены.")

except Exception as e:
    logging.critical(f"Критическая ошибка при инициализации сервисов Google Cloud: {e}", exc_info=True)
    raise RuntimeError(f"Не удалось инициализировать сервисы Google Cloud: {e}") from e

# --- Настройка CORS ---
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS")
origins = []
if CORS_ALLOWED_ORIGINS:
    origins.extend([origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",")])
else:
    logging.warning("Переменная окружения CORS_ALLOWED_ORIGINS не установлена. Используется список по умолчанию для разработки.")
    origins.extend([
        "https://ailbee.web.app", "https://ailbee.firebaseapp.com",
        "http://localhost", "http://localhost:3000", "http://localhost:5000", "http://localhost:8000"
    ])

logging.info(f"CORS настроен для следующих источников: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Обработчики исключений ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Произошла необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Произошла внутренняя ошибка сервера."}
    )

# --- Модели данных ---
class UserProfileUpdate(BaseModel):
    displayName: Optional[str] = None
    age: Optional[int] = None
    preferredLanguage: Optional[str] = None

class NewUser(BaseModel):
    email: str
    password: str
    displayName: str

class FCMToken(BaseModel):
    fcmToken: str = Field(..., description="Firebase Cloud Messaging registration token")

# ... (Остальные модели данных Pydantic остаются без изменений)

# --- Зависимость для аутентификации ---
async def get_current_user(request: Request):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Отсутствует заголовок авторизации")
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        logging.error(f"Ошибка верификации токена: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Недействительный токен: {e}")

# --- Вспомогательные функции ---
def convert_timestamps_to_strings(data: Any) -> Any:
    if isinstance(data, (firestore.Timestamp, datetime.datetime)):
        return data.isoformat()
    if isinstance(data, dict):
        return {k: convert_timestamps_to_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_timestamps_to_strings(elem) for elem in data]
    return data

# --- Эндпоинты ---
@app.get("/")
async def health_check():
    logging.info('Получен запрос на проверку здоровья (health check) /')
    return {"status": "ok", "message": "Service is healthy."}

@app.get("/api/")
async def root():
    logging.info('Получен запрос на /api/')
    return {"message": "Welcome to FastAPI Backend! Service is running."}

@app.post("/users/add")
async def add_user(new_user: NewUser):
    try:
        # URL вашей функции addUser Node.js
        url = "https://us-central1-ailbee.cloudfunctions.net/addUser"
        
        # Данные для отправки
        data = {
            "email": new_user.email,
            "password": new_user.password,
            "displayName": new_user.displayName
        }
        
        # Вызов функции Node.js
        response = requests.post(url, json=data)
        
        # Проверка ответа
        if response.status_code == 200:
            return JSONResponse(status_code=200, content={"status": "success", "message": "Пользователь успешно добавлен."})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logging.error(f"Ошибка при добавлении пользователя: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сервера при добавлении пользователя.")

@app.delete("/users/{user_id}")
async def delete_user_endpoint(user_id: str):
    try:
        # URL вашей функции deleteUser Node.js
        url = f"https://us-central1-ailbee.cloudfunctions.net/deleteUser/{user_id}"
        
        # Вызов функции Node.js
        response = requests.delete(url)
        
        # Проверка ответа
        if response.status_code == 200:
            return JSONResponse(status_code=200, content={"status": "success", "message": f"Пользователь {user_id} успешно удален."})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logging.error(f"Ошибка при удалении пользователя {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при удалении пользователя {user_id}.")

@app.get("/users/profile")
async def get_user_profile(user_id: str = Depends(get_current_user)):
    if db is None: raise HTTPException(status_code=503, detail="Сервис базы данных недоступен.")
    try:
        doc_ref = db.collection('users').document(user_id)
        doc = await asyncio.to_thread(doc_ref.get)
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Профиль пользователя не найден.")
        
        profile = convert_timestamps_to_strings(doc.to_dict())
        return JSONResponse(status_code=200, content={"status": "success", "profile": profile})
    except Exception as e:
        logging.error(f"Ошибка при получении профиля для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сервера при получении профиля.")

@app.put("/users/profile")
async def update_user_profile(profile_data: UserProfileUpdate, user_id: str = Depends(get_current_user)):
    if db is None: raise HTTPException(status_code=503, detail="Сервис базы данных недоступен.")
    
    update_data = profile_data.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="Нет данных для обновления.")
    
    try:
        user_ref = db.collection('users').document(user_id)
        await asyncio.to_thread(user_ref.update, update_data)
        logging.info(f"Профиль пользователя {user_id} обновлен.")
        
        updated_doc = await asyncio.to_thread(user_ref.get)
        updated_profile = convert_timestamps_to_strings(updated_doc.to_dict())
        
        return JSONResponse(status_code=200, content={"status": "success", "profile": updated_profile})
    except google_exceptions.NotFound:
        raise HTTPException(status_code=404, detail="Профиль пользователя не найден для обновления.")
    except Exception as e:
        logging.error(f"Ошибка при обновлении профиля для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка сервера при обновлении профиля.")

@app.post("/users/save-fcm-token")
async def save_fcm_token(token_data: FCMToken, user_id: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Сервис базы данных недоступен.")
    
    fcm_token = token_data.fcmToken
    if not fcm_token:
        raise HTTPException(status_code=400, detail="FCM токен не предоставлен.")
        
    try:
        # Сохраняем токен в подколлекции 'fcm_tokens' документа пользователя
        # Имя документа - сам токен, чтобы избежать дубликатов
        token_ref = db.collection('users').document(user_id).collection('fcm_tokens').document(fcm_token)
        
        # Сохраняем токен вместе с временной меткой
        await asyncio.to_thread(
            token_ref.set,
            {'updated_at': firestore.SERVER_TIMESTAMP},
            merge=True # Используем merge=True, чтобы обновить метку, если токен уже существует
        )
        
        logging.info(f"FCM токен успешно сохранен/обновлен для пользователя {user_id}")
        return JSONResponse(status_code=200, content={"status": "success", "message": "FCM токен успешно сохранен."})

    except Exception as e:
        logging.error(f"Ошибка при сохранении FCM токена для пользователя {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось сохранить FCM токен.")

@app.route("/library/books")
def list_books():
  # ... ваш существующий код для получения книг ...
  return {"books": []} # Просто пример

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")