# src/data_processing.py

import cv2
import os
import argparse
import glob

def extract_frames(video_path, output_folder, frame_skip):
    """
    Извлекает кадры из видео и сохраняет их в указанную папку.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Создана папка: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            image_name = f"video_{os.path.basename(video_path).split('.')[0]}_frame_{saved_count:06d}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Видео: {os.path.basename(video_path)}. Обработано {frame_count} кадров. Сохранено: {saved_count}.")

if __name__ == '__main__':
    # Настраиваем парсер аргументов для удобного запуска из командной строки
    parser = argparse.ArgumentParser(description="Инструмент для нарезки видео на кадры.")
    parser.add_argument('--video_dir', type=str, default='/app/data/01_raw', help='Папка с исходными видео.')
    parser.add_argument('--output_dir', type=str, default='/app/data/02_processed/frames', help='Папка для сохранения кадров.')
    parser.add_argument('--frame_skip', type=int, default=10, help='Сохранять каждый N-ный кадр.')
    
    args = parser.parse_args()

    video_files = glob.glob(os.path.join(args.video_dir, '*.mp4')) # Ищем все файлы .mp4
    
    if not video_files:
        print(f"В папке {args.video_dir} не найдено видеофайлов с расширением .mp4")
    else:
        print(f"Найдено видео для обработки: {len(video_files)}")
        for video_file in video_files:
            extract_frames(video_file, args.output_dir, args.frame_skip)
        print("Обработка всех видео завершена.")