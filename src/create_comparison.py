# src/create_comparison.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)

import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def create_comparison_image(model, video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Ошибка: Не удалось прочитать кадр №{frame_number}")
        cap.release()
        return None
    cap.release()
    
    print(f"Кадр №{frame_number} успешно извлечен. Начинаем обработку...")

    resolutions = [1280, 640, 320]
    processed_images = []

    for res in resolutions:
        print(f"Обработка в разрешении {res}px...")
        results = model(frame, imgsz=res, conf=0.5, verbose=False)
        annotated_frame = results[0].plot(line_width=2)
        
        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
        # Теперь мы изменяем размер самого изображения, чтобы визуализировать разницу
        h, w, _ = annotated_frame.shape
        # Сохраняем пропорции, вписывая изображение в квадрат res x res
        scale = res / max(h, w)
        display_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        
        # Добавляем подпись
        label = f"Processed at: {res}px"
        cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed_images.append(display_frame)

    # --- Сборка умного коллажа, так как картинки теперь разного размера ---
    # Находим максимальную высоту и общую ширину
    max_height = max(img.shape[0] for img in processed_images)
    total_width = sum(img.shape[1] for img in processed_images)
    
    # Создаем черный холст
    canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    
    # Размещаем каждую картинку на холсте
    current_x = 0
    for img in processed_images:
        h, w, _ = img.shape
        canvas[0:h, current_x:current_x + w] = img
        current_x += w
        
    print("Коллаж для сравнения успешно создан.")
    return canvas

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для создания сравнительного изображения.")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к обученной модели (.pt).')
    parser.add_argument('--source_video', type=str, default='/app/test_video.mp4', help='Путь к исходному видеофайлу внутри контейнера.')
    parser.add_argument('--frame_num', type=int, required=True, help='Номер кадра для анализа.')
    parser.add_argument('--output_path', type=str, default='/app/data/comparison.jpg', help='Путь для сохранения итогового изображения.')
    
    args = parser.parse_args()

    model = YOLO(args.model_path)
    final_image = create_comparison_image(model, args.source_video, args.frame_num)

    if final_image is not None:
        cv2.imwrite(args.output_path, final_image)
        print(f"Результат сохранен в {args.output_path}")