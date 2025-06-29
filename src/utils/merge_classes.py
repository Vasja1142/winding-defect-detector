# src/utils/merge_classes.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def merge_classes_to_one(input_dir, output_dir, new_class_id=0):
    """
    Проходит по всем .txt файлам разметки и заменяет
    индекс любого класса на единый new_class_id.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_path.glob('*.txt'))
    print(f"Найдено {len(label_files)} файлов разметки для обработки.")

    for label_file in tqdm(label_files, desc="Объединение классов"):
        new_lines = []
        lines = label_file.read_text().splitlines()
        
        for line in lines:
            parts = line.split()
            if not parts: continue
            # Заменяем первый элемент (индекс класса) на new_class_id
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts))
        
        # Сохраняем измененный файл в новую директорию
        (output_path / label_file.name).write_text("\n".join(new_lines))
        
        # Копируем соответствующее изображение
        image_file = label_file.with_suffix('.jpg')
        if image_file.exists():
            os.system(f'cp "{image_file}" "{output_path / image_file.name}"')

    print(f"\n--- Объединение завершено! ---")
    print(f"Новый датасет с одним классом (ID={new_class_id}) сохранен в: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для объединения всех классов YOLO в один.")
    parser.add_argument('--input_dir', required=True, type=str, help='Папка с исходным многоклассовым датасетом.')
    parser.add_argument('--output_dir', required=True, type=str, help='Папка для сохранения нового одноклассового датасета.')
    
    args = parser.parse_args()
    merge_classes_to_one(args.input_dir, args.output_dir)