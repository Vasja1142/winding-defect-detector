# src/create_verification_set.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)
import os
import argparse
import shutil
import sys
from tqdm import tqdm

def create_balanced_set(input_dir, output_dir, neg_sample_rate):
    """
    Создает сбалансированный набор данных для верификации.
    Копирует все изображения с аннотациями (позитивные) и
    каждое N-е изображение без аннотации (негативные).
    
    ИСПРАВЛЕНИЕ: Для негативных примеров создается ПУСТОЙ .txt файл.
    """
    if not os.path.isdir(input_dir):
        print(f"[ОШИБКА] Директория с исходными данными не найдена: {input_dir}", file=sys.stderr)
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    all_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])
    
    print(f"Найдено {len(all_images)} изображений. Начинаем сэмплирование...")
    
    positive_count = 0
    negative_sampled_count = 0
    negative_counter = 0

    for image_name in tqdm(all_images, desc="Обработка кадров"):
        base_name = os.path.splitext(image_name)[0]
        label_name = f"{base_name}.txt"
        
        image_path = os.path.join(input_dir, image_name)
        label_path = os.path.join(input_dir, label_name)
        
        # Проверяем, есть ли разметка (позитивный пример)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            shutil.copy(image_path, os.path.join(output_dir, image_name))
            shutil.copy(label_path, os.path.join(output_dir, label_name))
            positive_count += 1
        else:
            # Это негативный пример. Берем каждого N-го.
            if negative_counter % neg_sample_rate == 0:
                # Копируем само изображение
                shutil.copy(image_path, os.path.join(output_dir, image_name))
                
                # *** ГЛАВНОЕ ИЗМЕНЕНИЕ: СОЗДАЕМ ПУСТОЙ .txt ФАЙЛ ***
                output_label_path = os.path.join(output_dir, label_name)
                open(output_label_path, 'a').close()
                # ****************************************************
                
                negative_sampled_count += 1
            negative_counter += 1
            
    print("\n--- Сборка Верификационного Набора Завершена! ---")
    print(f"Скопировано позитивных (с дефектами) кадров: {positive_count}")
    print(f"Скопировано негативных (без дефектов) кадров: {negative_sampled_count}")
    print(f"Всего кадров в 'золотом' наборе: {positive_count + negative_sampled_count}")
    print(f"Результат находится в папке: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Сборка сбалансированного датасета для верификации.")
    parser.add_argument('--input_dir', required=True, type=str, help='Папка с экспортом из CVAT (где все jpg и txt).')
    parser.add_argument('--output_dir', required=True, type=str, help='Папка для сохранения сбалансированного набора.')
    parser.add_argument('--neg_sample_rate', type=int, default=20, help='Брать каждый N-й кадр без дефектов.')
    
    args = parser.parse_args()
    create_balanced_set(args.input_dir, args.output_dir, args.neg_sample_rate)