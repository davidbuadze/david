# Используем официальный образ Python как базовый образ
FROM python:3.10-slim-buster

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения
COPY . .

# Устанавливаем переменную окружения для порта, которую Cloud Run будет использовать
EXPOSE 8080

CMD ["python", "main.py"]
