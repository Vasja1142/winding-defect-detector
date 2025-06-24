# src/analyze_dataset.py

import os
import glob
import argparse

def analyze_subset(subset_path):
    """
    Анализирует под-выборку (train или valid) и возвращает количество
    изображений с дефектами и без.
    """
    labels_path = os.path.join(subset_path, 'labels')
    if not os.path.isdir(labels_path):
        return 0, 0

    defect_count = 0
    no_defect_count = 0
    
    label_files = glob.glob(os.path.join(labels_path, '*.txt'))
    
    for label_file in label_files:
        if os.path.getsize(label_file) > 0:
            defect_count += 1
        else:
            no_defect_count += 1
            
    return defect_count, no_defect_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Анализатор датасета YOLO.")
    parser.add_argument('--dataset_dir', type=str, default='/app/data/03_annotated', 
                        help='Путь к папке с финальным датасетом (содержащей train и valid).')
    args = parser.parse_args()

    train_path = os.path.join(args.dataset_dir, 'train')
    valid_path = os.path.join(args.dataset_dir, 'valid')

    train_defects, train_no_defects = analyze_subset(train_path)
    valid_defects, valid_no_defects = analyze_subset(valid_path)

    total_defects = train_defects + valid_defects
    total_no_defects = train_no_defects + valid_no_defects
    total_images = total_defects + total_no_defects

    print("=" * 40)
    print("АНАЛИЗ ФИНАЛЬНОГО ДАТАСЕТА")
    print("=" * 40)
    
    if total_images == 0:
        print("Датасет пуст или не найден. Проверьте путь.")
    else:
        print(f"Обучающая выборка (Train):")
        print(f"  - С дефектами:    {train_defects}")
        print(f"  - Без дефектов:   {train_no_defects}")
        print(f"  - Всего:          {train_defects + train_no_defects}")
        print("-" * 40)
        print(f"Валидационная выборка (Valid):")
        print(f"  - С дефектами:    {valid_defects}")
        print(f"  - Без дефектов:   {valid_no_defects}")
        print(f"  - Всего:          {valid_defects + valid_no_defects}")
        print("=" * 40)
        print(f"ИТОГО ПО ДАТАСЕТУ:")
        print(f"  - Всего изображений с дефектами: {total_defects} ({total_defects / total_images:.1%})")
        print(f"  - Всего изображений без дефектов: {total_no_defects} ({total_no_defects / total_images:.1%})")
        print(f"  - Общее количество изображений:   {total_images}")
        print("=" * 40)