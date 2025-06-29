# src/create_labeled_video.py
import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def process_video(model_path: str, input_video: str, output_video: str, conf_threshold: float, imgsz: int):
    """
    Обрабатывает входное видео, наносит на него рамки детекции и сохраняет в новый файл.
    """
    # --- 1. Загрузка модели ---
    print(f"Загрузка модели из: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    # --- 2. Открытие видеофайлов ---
    input_path = Path(input_video)
    output_path = Path(output_video)
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть исходное видео: {input_path}")
        return

    # Получаем свойства видео для создания выходного файла
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создаем объект для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Кодек для .mp4 файлов
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    print(f"Обработка видео: {input_video}")
    print(f"Результат будет сохранен в: {output_video}")

    # --- 3. По-кадровая обработка ---
    with tqdm(total=total_frames, desc="Создание видео") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Выполняем детекцию на кадре
            results = model(frame, imgsz=imgsz, conf=conf_threshold, verbose=False)
            
            # Используем встроенный метод .plot() для отрисовки рамок и меток
            annotated_frame = results[0].plot()

            # Записываем аннотированный кадр в выходной файл
            writer.write(annotated_frame)
            
            pbar.update(1)

    # --- 4. Очистка ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("\n--- Готово! Видео с метками успешно создано. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для создания видео с нанесенными метками детекции.")
    parser.add_argument('--model_path', required=True, type=str, help='Путь к обученной модели .pt.')
    parser.add_argument('--input_video', required=True, type=str, help='Путь к исходному видеофайлу.')
    parser.add_argument('--output_video', required=True, type=str, help='Путь для сохранения итогового видеофайла.')
    parser.add_argument('--conf', type=float, default=0.5, help='Порог уверенности для детекции.')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения для инференса.')
    
    args = parser.parse_args()
    
    process_video(args.model_path, args.input_video, args.output_video, args.conf, args.imgsz)