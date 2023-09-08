# Используйте официальный образ Python
FROM python:3.8

# Устанавливаем рабочую директорию
WORKDIR /usr/src/app

# Установка зависимостей
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы вашего приложения в контейнер
COPY . .

# Указываем команду для запуска приложения
CMD [ "python", "./Tinkoff_Sirius_bot.py" ]