# src/data_processing.py (МОДЕРНИЗИРОВАННАЯ ВЕРСИЯ)
import cv2
import os
import argparse
from tqdm import tqdm

def process_single_video(video_path, output_dir, frame_skip):
    """
    Обрабатывает один видеофайл, нарезая его на кадры.
    Имена кадров содержат префикс из имени видео.
    """
    video_filename = os.path.basename(video_path)
    video_name_prefix = os.path.splitext(video_filename)[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Предупреждение: Не удалось открыть видеофайл: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0

    # Создаем прогресс-бар для текущего видео
    pbar = tqdm(total=total_frames, desc=f"Обработка {video_filename}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        if frame_count % frame_skip == 0:
            # Формируем имя файла с префиксом
            image_name = f"{video_name_prefix}_frame_{saved_count:06d}.jpg"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, frame)
            saved_count += 1
        
        frame_count += 1

    pbar.close()
    cap.release()
    print(f"-> Завершено. Сохранено {saved_count} кадров из {video_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Нарезка видео на кадры.")
    # Сделали аргументы взаимоисключающими. Теперь можно передать или папку, или файл.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video_dir', type=str, help='Папка с видеофайлами для обработки.')
    group.add_argument('--video_file', type=str, help='Путь к одному видеофайлу для обработки.')
    
    parser.add_argument('--output_dir', required=True, type=str, help='Папка для сохранения кадров.')
    parser.add_argument('--frame_skip', type=int, default=4, help='Сохранять каждый N-ный кадр.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.video_dir:
        # Старый режим: обработка всех видео в папке
        video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        for video_path in video_files:
            process_single_video(video_path, args.output_dir, args.frame_skip)
    elif args.video_file:
        # Новый режим: обработка только одного указанного файла
        process_single_video(args.video_file, args.output_dir, args.frame_skip)