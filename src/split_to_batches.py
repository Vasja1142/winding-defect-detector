# src/split_to_batches.py
import os
import argparse
import shutil
import sys

def split_files_to_batches(source_dir, dest_dir, batch_size):
    """
    Разделяет файлы из исходной директории на несколько поддиректорий (батчей).
    """
    if not os.path.isdir(source_dir):
        print(f"[ОШИБКА] Исходная директория не найдена: {source_dir}", file=sys.stderr)
        sys.exit(1)
        
    os.makedirs(dest_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    files.sort() # Сортируем для последовательности
    
    print(f"Найдено {len(files)} файлов для разделения на батчи размером {batch_size}.")
    
    batch_num = 0
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_dir_name = f"batch_{batch_num:03d}"
        batch_full_path = os.path.join(dest_dir, batch_dir_name)
        
        os.makedirs(batch_full_path, exist_ok=True)
        
        print(f"  -> Создание {batch_dir_name} ({len(batch_files)} файлов)...")
        
        for filename in batch_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(batch_full_path, filename))
            
        batch_num += 1
        
    print(f"\nРазделение завершено. Создано {batch_num} батчей в директории: {dest_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Разделение набора файлов на батчи.")
    parser.add_argument('--source_dir', required=True, type=str, help='Исходная папка с файлами.')
    parser.add_argument('--dest_dir', required=True, type=str, help='Папка, куда будут сложены батчи.')
    parser.add_argument('--batch_size', type=int, default=2000, help='Количество файлов в одном батче.')
    
    args = parser.parse_args()
    
    split_files_to_batches(args.source_dir, args.dest_dir, args.batch_size)