# src/verify_filenames.py
import os
import argparse

def verify_matching_names(directory):
    images = {os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith('.jpg')}
    labels = {os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith('.txt')}
    
    missing_labels = images - labels
    missing_images = labels - images
    
    print("--- Запуск Проверки Соответствия Имен Файлов ---")
    
    if not missing_labels and not missing_images:
        print(f"[OK] Все {len(images)} изображений и {len(labels)} меток полностью соответствуют друг другу.")
        return True

    if missing_labels:
        print(f"\n[ОШИБКА] Найдено {len(missing_labels)} изображений, для которых НЕТ файла разметки:")
        for name in sorted(list(missing_labels))[:10]: print(f"  - {name}.jpg")

    if missing_images:
        print(f"\n[ОШИБКА] Найдено {len(missing_images)} файлов разметки, для которых НЕТ изображения:")
        for name in sorted(list(missing_images))[:10]: print(f"  - {name}.txt")
        
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Директория для проверки')
    args = parser.parse_args()
    verify_matching_names(args.dir)