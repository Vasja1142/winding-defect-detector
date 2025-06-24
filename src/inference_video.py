# src/inference_video.py (ПРАВИЛЬНАЯ ВЕРСИЯ)

import cv2
import os
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def process_video(model, source_video_path, output_dir, confidence_threshold):
    """
    Обрабатывает видео с помощью модели YOLO и сохраняет результат.
    """
    if not os.path.exists(source_video_path):
        print(f"Ошибка: Видеофайл не найден по пути {source_video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {source_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = os.path.basename(source_video_path).replace('.mp4', '_annotated.mp4')
    output_video_path = os.path.join(output_dir, output_filename)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print(f"Обработка видео: {source_video_path}")
    print(f"Результат будет сохранен в: {output_video_path}")

    for _ in tqdm(range(total_frames), desc="Обработка кадров"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()
    print("Обработка завершена!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для применения модели YOLO к видео.")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к обученной модели (.pt).')
    parser.add_argument('--source', type=str, required=True, help='Путь к исходному видеофайлу.')
    parser.add_argument('--output_dir', type=str, default='/app/data/06_inference_results', help='Папка для сохранения обработанного видео.')
    parser.add_argument('--confidence', type=float, default=0.6, help='Порог уверенности для детекции.')
    
    args = parser.parse_args()

    model = YOLO(args.model_path)
    process_video(model, args.source, args.output_dir, args.confidence)