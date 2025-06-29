# src/utils/augment_offline.py (ВЕРСИЯ С РУЧНЫМ УПРАВЛЕНИЕМ ШУМОМ)
import os
import cv2
import numpy as np
import argparse
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import random

def add_manual_gauss_noise(image, sigma):
    """
    Добавляет Гауссов шум к изображению с заданным стандартным отклонением (sigma).
    Работает с uint8 изображениями.
    """
    # Создаем Гауссов шум с mean=0 и нашим sigma
    gauss = np.random.normal(0, sigma, image.shape).astype('float32')
    # Добавляем шум к изображению
    noisy_image = cv2.add(image.astype('float32'), gauss)
    # Обрезаем значения, чтобы они остались в диапазоне [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype('uint8')


def augment_dataset_offline(input_dir, output_dir, noise_sigma, noise_probability):
    """
    Создает аугментированные копии изображений, добавляя размытие и
    полностью контролируемый Гауссов шум.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Конвейер аугментаций ТОЛЬКО для размытия ---
    blur_transform = A.Compose([
        A.OneOf([
            A.GaussianBlur(p=0.8, blur_limit=(3, 9)),
            A.MedianBlur(p=0.8, blur_limit=(3, 13)),
        ], p=0.8), # с вероятностью 80% применяем один из видов размытия
        
        A.OneOf([
            A.ISONoise(p=0.6, color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.8), # с вероятностью 80% применяем один из видов шума
    ])
    image_files = list(input_path.glob('*.jpg'))
    print(f"Найдено {len(image_files)} изображений для обработки.")
    
    for image_file in tqdm(image_files, desc="Аугментация (Blur/Noise)"):
        label_file = image_file.with_suffix('.txt')
        
        if label_file.exists() and os.path.getsize(str(label_file)) >= 0:
            image = cv2.imread(str(image_file))
            
            # --- ШАГ 1: Применяем размытие (через Albumentations) ---
            transformed = blur_transform(image=image)
            processed_image = transformed['image']
            
            # --- ШАГ 2: Применяем наш ручной шум (через нашу функцию) ---
            if random.random() < noise_probability:
                processed_image = add_manual_gauss_noise(processed_image, sigma=noise_sigma)
            
            # Сохраняем итоговое изображение
            new_image_name = f"aug_{image_file.name}"
            cv2.imwrite(str(output_path / new_image_name), processed_image)
            
            # Копируем для него оригинальный файл разметки
            os.system(f'cp "{label_file}" "{output_path / new_image_name.replace(".jpg", ".txt")}"')

    print(f"\n--- Офлайн-аугментация завершена! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    # --- Добавляем параметры для тонкой настройки в командную строку ---
    parser.add_argument('--sigma', type=float, default=5.0, help='Сила шума (стандартное отклонение). Рекомендую начать с 2.0-10.0')
    parser.add_argument('--noise_p', type=float, default=0.9, help='Вероятность добавления шума к изображению (0.0 до 1.0).')
    
    args = parser.parse_args()
    augment_dataset_offline(args.input_dir, args.output_dir, args.sigma, args.noise_p)