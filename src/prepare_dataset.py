# src/prepare_dataset.py (v4 - Финальная, упрощенная версия)

import os
import glob
import shutil
import random
import argparse
from tqdm import tqdm

def create_dirs(base_path):
    """Создает правильную структуру папок для YOLO."""
    os.makedirs(os.path.join(base_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'valid', 'labels'), exist_ok=True)

def copy_files(basenames, image_source_dir, label_source_dir, dest_dir_base):
    """Копирует пары .jpg и .txt файлов в целевую директорию."""
    image_dest = os.path.join(dest_dir_base, 'images')
    label_dest = os.path.join(dest_dir_base, 'labels')
    
    for basename in tqdm(basenames, desc=f"Копирование в {dest_dir_base}"):
        shutil.copy(os.path.join(image_source_dir, basename + '.jpg'), image_dest)
        shutil.copy(os.path.join(label_source_dir, basename + '.txt'), label_dest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Финальная сборка датасета из верифицированной разметки.")
    parser.add_argument('--image_dir', type=str, required=True, help='Папка со ВСЕМИ исходными изображениями (e.g., /app/data/02_processed/frames).')
    parser.add_argument('--label_dir', type=str, required=True, help='Папка с ВЕРИФИЦИРОВАННОЙ разметкой (e.g., /app/data/temp_cvat_verified_export/obj_train_data).')
    parser.add_argument('--output_dir', type=str, default='/app/data/03_annotated', help='Финальная папка для датасета.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Доля данных для валидации.')
    
    args = parser.parse_args()

    # Удаляем старую папку для чистоты
    if os.path.exists(args.output_dir):
        print(f"Очистка старой директории: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    create_dirs(args.output_dir)

    # --- Ключевое изменение: мы больше не фильтруем и не прореживаем! ---
    # Мы просто берем ВСЕ файлы из папки с верифицированной разметкой.
    print(f"Чтение верифицированной разметки из: {args.label_dir}")
    label_files = glob.glob(os.path.join(args.label_dir, '*.txt'))
    
    if not label_files:
        print("Ошибка: В указанной папке с разметкой не найдено .txt файлов. Проверьте путь.")
        exit()

    final_basenames = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
    
    # Перемешиваем и делим на train/valid
    random.shuffle(final_basenames)
    split_index = int(len(final_basenames) * (1 - args.val_split))
    
    train_basenames = final_basenames[:split_index]
    valid_basenames = final_basenames[split_index:]

    print("-" * 30)
    print(f"Итого в датасете: {len(final_basenames)} изображений.")
    print(f"Обучающая выборка: {len(train_basenames)} изображений.")
    print(f"Валидационная выборка: {len(valid_basenames)} изображений.")
    print("-" * 30)

    # Копируем файлы в финальные директории
    copy_files(train_basenames, args.image_dir, args.label_dir, os.path.join(args.output_dir, 'train'))
    copy_files(valid_basenames, args.image_dir, args.label_dir, os.path.join(args.output_dir, 'valid'))

    print("Сборка финального датасета успешно завершена!")
    print(f"Готовый датасет находится в папке: {args.output_dir}")