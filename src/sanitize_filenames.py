# src/sanitize_filenames.py (ПРАВИЛЬНАЯ, УНИВЕРСАЛЬНАЯ ВЕРСИЯ)
import os
import argparse
from unidecode import unidecode
import re
import sys

def sanitize_name(filename):
    """Приводит имя файла к безопасному формату: латиница, нижний регистр, с подчеркиваниями вместо пробелов."""
    base_name, extension = os.path.splitext(filename)
    
    # Транслитерация, нижний регистр, замена не-букв/цифр на подчеркивание
    sanitized_base = unidecode(base_name).lower()
    sanitized_base = re.sub(r'[\s\W]+', '_', sanitized_base).strip('_')
    sanitized_base = re.sub(r'__+', '_', sanitized_base)
    
    # Расширение тоже приводим к нижнему регистру
    sanitized_ext = extension.lower()
    
    return f"{sanitized_base}{sanitized_ext}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Универсальный санитайзер имен файлов в директории.")
    parser.add_argument('--input_dir', required=True, type=str, help='Директория, в которой нужно переименовать файлы.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ОШИБКА] Директория не найдена: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Сканирование директории для санитарной обработки: {args.input_dir}")
    
    # Получаем список файлов ПЕРЕД переименованием
    for filename in os.listdir(args.input_dir):
        original_path = os.path.join(args.input_dir, filename)
        
        # Пропускаем папки, если они вдруг есть
        if os.path.isdir(original_path):
            continue
            
        sanitized_filename = sanitize_name(filename)
        
        if filename != sanitized_filename:
            new_path = os.path.join(args.input_dir, sanitized_filename)
            try:
                os.rename(original_path, new_path)
                print(f"  -> Переименовано: '{filename}' -> '{sanitized_filename}'")
            except OSError as e:
                print(f"[ОШИБКА] не удалось переименовать '{filename}': {e}", file=sys.stderr)
        
    print("\nСанитарная обработка имен файлов завершена!")