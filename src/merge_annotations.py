# src/merge_annotations.py
import os
import shutil
import argparse
from tqdm import tqdm

def merge(original_dir, verified_dir, output_dir):
    """
    Сливает две папки с разметкой, отдавая приоритет 'verified_dir'.
    """
    print(f"Копирование оригинальной разметки из {original_dir} в {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(original_dir, output_dir)
    
    print(f"Обновление разметки из {verified_dir}...")
    verified_labels_dir = os.path.join(verified_dir, 'obj_train_data')
    if not os.path.exists(verified_labels_dir):
        print(f"Папка с верифицированной разметкой не найдена: {verified_labels_dir}")
        return

    for filename in tqdm(os.listdir(verified_labels_dir)):
        shutil.copy(
            os.path.join(verified_labels_dir, filename),
            os.path.join(output_dir, 'obj_train_data', filename)
        )
    print("Слияние завершено.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Слияние оригинальной и верифицированной разметки.")
    parser.add_argument('--original_dir', type=str, default='/app/data/temp_cvat_export', help='Папка с оригинальным экспортом CVAT.')
    parser.add_argument('--verified_dir', type=str, required=True, help='Папка с экспортом из верификационной задачи.')
    parser.add_argument('--output_dir', type=str, default='/app/data/merged_labels', help='Выходная папка для объединенной разметки.')
    args = parser.parse_args()
    merge(args.original_dir, args.verified_dir, args.output_dir)