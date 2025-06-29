# src/run_data_prep.py (ФИНАЛЬНАЯ ВЕРСИЯ)
import subprocess
import os
import sys

# --- КОНФИГУРАЦИЯ ---
RAW_VIDEO_DIR = "/app/data/01_raw"
PROCESSED_FRAMES_DIR = "/app/data/02_processed/frames"
FRAME_SKIP = 4 # Сохраняем каждый 4-й кадр

# Определяем видео, которое нужно ИСКЛЮЧИТЬ из обработки
# Указываем его САНИРОВАННОЕ имя, так как этот этап идет после переименования
TEST_VIDEO_SANITIZED_NAME = "nru_2025_06_16_12_49_50.mp4"

os.makedirs(PROCESSED_FRAMES_DIR, exist_ok=True)

print("--- Запуск Пайплайна Подготовки Данных ---")

try:
    # --- ЭТАП 1: Санитарная обработка имен ВСЕХ файлов ---
    print(f"\n[ЭТАП 1/3] Запуск санитайзера для папки: {RAW_VIDEO_DIR}")
    subprocess.run(["python3", "src/sanitize_filenames.py", "--input_dir", RAW_VIDEO_DIR], check=True)
    print("[УСПЕХ] Имена файлов успешно санированы.")

    # --- ЭТАП 2: Определение списка видео для обучения ---
    print(f"\n[ЭТАП 2/3] Определение списка обучающих видео...")
    all_videos = [f for f in os.listdir(RAW_VIDEO_DIR) if f.lower().endswith('.mp4')]
    
    if TEST_VIDEO_SANITIZED_NAME not in all_videos:
        print(f"[ОШИБКА] Тестовое видео '{TEST_VIDEO_SANITIZED_NAME}' не найдено в папке {RAW_VIDEO_DIR}")
        sys.exit(1)

    train_videos = [v for v in all_videos if v != TEST_VIDEO_SANITIZED_NAME]
    
    print(f"-> Найдено видео для обучения: {len(train_videos)} шт.")
    for video in train_videos:
        print(f"  - {video}")
    print(f"-> Видео для тестирования (пропущено): {TEST_VIDEO_SANITIZED_NAME}")

    # --- ЭТАП 3: Поочередная нарезка ТОЛЬКО обучающих видео ---
    print(f"\n[ЭТАП 3/3] Запуск нарезки кадров. Сохраняется каждый {FRAME_SKIP}-й кадр.")
    for video_file in train_videos:
        video_path = os.path.join(RAW_VIDEO_DIR, video_file)
        subprocess.run([
            "python3", "src/data_processing.py",
            "--video_file", video_path,
            "--output_dir", PROCESSED_FRAMES_DIR,
            "--frame_skip", str(FRAME_SKIP)
        ], check=True)
    
    print("\n--- Пайплайн Подготовки Данных Успешно Завершен! ---")
    print(f"В папке {PROCESSED_FRAMES_DIR} теперь находятся кадры ТОЛЬКО из обучающих видео.")

except subprocess.CalledProcessError as e:
    print(f"\n[ОШИБКА] Один из этапов завершился с ошибкой: {e}", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError as e:
    print(f"\n[ОШИБКА] Файл не найден: {e}", file=sys.stderr)
    sys.exit(1)