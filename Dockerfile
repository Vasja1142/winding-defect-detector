# Используем Python 3.12 как стабильный и проверенный стандарт
FROM python:3.12-slim

# Устанавливаем системные зависимости, необходимые для OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
# Устанавливаем самую свежую Nightly-сборку PyTorch напрямую
# Она содержит поддержку новейших GPU
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# Копируем файл с остальными зависимостями
COPY requirements.txt .
# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код нашего приложения
COPY ./src ./src
COPY ./config.yaml .

# COPY ./data/01_raw/source_video.mp4 /app/test_video.mp4
# Команда по умолчанию при запуске контейнера
CMD ["python", "src/inference.py"]