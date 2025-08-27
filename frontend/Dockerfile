# Stage 1: Build the React application
FROM node:22-alpine as builder # Используем node:22-alpine для свежей среды

WORKDIR /app # Рабочая директория внутри контейнера

# Копируем package.json и package-lock.json для установки зависимостей
# Это позволяет Docker кэшировать слой установки зависимостей
COPY package.json ./
COPY package-lock.json ./ # Убедитесь, что у вас есть package-lock.json

# Устанавливаем зависимости (включая devDependencies для сборки React)
RUN npm install

# Копируем остальной код приложения (src, public и т.д.)
COPY . ./

# Собираем React-приложение для продакшена
# Результат будет в папке /app/build
RUN npm run public

# Этот Dockerfile предназначен для сборки папки 'public'.
# Мы не будем запускать этот образ как сервер.
# Вместо этого, Cloud Build извлечет папку 'build' из этого образа
# и загрузит ее в Cloud Storage для Firebase Hosting.
