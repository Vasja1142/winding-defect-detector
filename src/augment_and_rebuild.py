# src/augment_and_rebuild.py

import cv2
import os
import glob
import argparse
import albumentations as A
from tqdm import tqdm

def read_yolo_labels(label_path):
    if not os.path.exists(label_path): return []
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
        return [[int(l[0])] + list(map(float, l[1:])) for l in labels]

def write_yolo_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for label in labels:
            class_idx, x, y, w, h = label
            f.write(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Универсальный скрипт для аугментации дефектов в датасете.")
    parser.add_argument('--dataset_dir', default='/app/data/03_annotated', help='Основная папка датасета (с train/valid).')
    parser.add_argument('--num_copies', type=int, default=15, help='Сколько аугментированных копий создавать с каждого дефекта.')
    args = parser.parse_args()

    # 1. Определяем нашу цепочку аугментаций
    transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomScale(scale_limit=(0.0, 0.4), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    # 2. Находим все изображения с дефектами в датасете
    defect_files = []
    for subset in ['train', 'valid']:
        subset_path = os.path.join(args.dataset_dir, subset)
        for label_path in glob.glob(os.path.join(subset_path, 'labels', '*.txt')):
            if os.path.getsize(label_path) > 0:
                image_path = label_path.replace('/labels/', '/images/').replace('.txt', '.jpg')
                if os.path.exists(image_path):
                    defect_files.append((image_path, label_path))
    
    if not defect_files:
        print("Ошибка: Не найдено ни одного изображения с дефектами в датасете.")
        exit()

    print(f"Найдено {len(defect_files)} оригинальных изображений с дефектами. Начинаем аугментацию...")

    # 3. Аугментируем и сохраняем НАПРЯМУЮ в папку train
    output_images_dir = os.path.join(args.dataset_dir, 'train', 'images')
    output_labels_dir = os.path.join(args.dataset_dir, 'train', 'labels')
    
    new_files_count = 0
    for image_path, label_path in tqdm(defect_files, desc="Аугментация дефектов"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = read_yolo_labels(label_path)
        coords = [b[1:] for b in bboxes]
        class_labels = [b[0] for b in bboxes]

        for i in range(args.num_copies):
            augmented = transform(image=image, bboxes=coords, class_labels=class_labels)
            if not augmented['bboxes']: continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Генерируем уникальное имя для аугментированного файла
            aug_image_name = f"{base_name}_aug_{i}.jpg"
            aug_label_name = f"{base_name}_aug_{i}.txt"
            
            # Сохраняем аугментированное изображение
            cv2.imwrite(os.path.join(output_images_dir, aug_image_name), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
            
            # Сохраняем аугментированную разметку
            yolo_labels = [[cls] + list(bbox) for bbox, cls in zip(augmented['bboxes'], augmented['class_labels'])]
            write_yolo_labels(os.path.join(output_labels_dir, aug_label_name), yolo_labels)
            new_files_count += 1

    print("=" * 30)
    print(f"Аугментация завершена. Добавлено {new_files_count} новых изображений.")
    print("=" * 30)