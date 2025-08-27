    # Используем официальный образ Python как базовый
    FROM python:3.10-slim

    # Устанавливаем рабочий каталог в контейнере
    WORKDIR /app

    # Копируем файл зависимостей
    COPY requirements.txt ./

    # Устанавливаем зависимости из requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    # Копируем исходный код приложения
    COPY . .

    # Команда для запуска приложения
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
    