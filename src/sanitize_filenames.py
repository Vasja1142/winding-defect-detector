# src/sanitize_filenames.py

import os
import argparse
from unidecode import unidecode
from tqdm import tqdm

def sanitize(filename):
    """Преобразует имя файла в 'безопасный' ASCII формат."""
    # Транслитерируем с помощью Unidecode
    safe_name = unidecode(filename)
    # Заменяем пробелы и другие нежелательные символы на '_'
    safe_name = safe_name.replace(' ', '_').replace('-', '_').replace(':', '_')
    return safe_name

def process_directory(directory_path):
    """Переименовывает все файлы в указанной директории."""
    if not os.path.isdir(directory_path):
        print(f"Директория не найдена: {directory_path}")
        return
        
    print(f"Обработка директории: {directory_path}")
    # Мы читаем список файлов один раз, чтобы избежать проблем при переименовании
    filenames = os.listdir(directory_path)
    for filename in tqdm(filenames):
        old_path = os.path.join(directory_path, filename)
        # Проверяем, что это файл, а не папка
        if os.path.isfile(old_path):
            new_filename = sanitize(filename)
            new_path = os.path.join(directory_path, new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Санитайзер имен файлов в датасете.")
    parser.add_argument('--input_dir', type=str, required=True, help='Папка с распакованными данными (например, temp_cvat_export).')
    
    args = parser.parse_args()
    
    # Обрабатываем папки images и labels
    process_directory(os.path.join(args.input_dir, 'images'))
    process_directory(os.path.join(args.input_dir, 'labels'))
    
    print("Санитарная обработка имен файлов завершена!")