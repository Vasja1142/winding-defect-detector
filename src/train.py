# src/train.py
import os
import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для обучения модели YOLO.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к файлу конфигурации датасета.')
    parser.add_argument('--model', type=str, default='yolo12n.pt', help='Предобученная модель YOLO для старта (n, s, m, l, x).')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох обучения.')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча. Уменьшите, если не хватает VRAM.')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения для обучения.')
    parser.add_argument('--project', type=str, default='/app/data/05_training_runs', help='Папка для сохранения результатов обучения.')
    
    args = parser.parse_args()

    # Собираем полный путь к модели внутри контейнера
    model_path = os.path.join('/app/data/00_pretrained_models', args.model)
    model = YOLO(model_path)

    print("="*30)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ (Демо-версия)")
    print(f"Конфиг: {args.config}")
    print(f"Модель: {args.model}")
    print(f"Эпохи: {args.epochs}")
    print(f"Размер батча: {args.batch}")
    print(f"Папка результатов: {args.project}")
    print("="*30)
    
    # Запускаем обучение
    model.train(
        data=args.config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name='demo_m_model_v0.4' # Дадим имя нашему первому демо-запуску
    )

    print("Обучение завершено!")