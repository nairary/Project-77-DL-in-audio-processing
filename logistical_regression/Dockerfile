FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Указываем порт, который будет слушать приложение
EXPOSE 8000

# Запускаем приложение
CMD ["python", "model_trainer/main.py"]