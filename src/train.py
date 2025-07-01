# src/train.py (Версия с интеграцией MLflow)
import argparse
import os
import mlflow
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для обучения и дообучения YOLO с логированием в MLflow.")
    
    # Аргументы для модели и данных
    parser.add_argument('--model', required=True, type=str, help='Путь к модели для старта (yolov8s.pt) или для продолжения (.../last.pt).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к файлу конфигурации датасета.')
    
    # Аргументы для MLflow
    parser.add_argument('--mlflow_experiment_name', type=str, default='Winding Defect Detection', help='Имя эксперимента в MLflow.')
    
    # Основные параметры обучения
    parser.add_argument('--epochs', type=int, default=150, help='Количество эпох обучения.')
    parser.add_argument('--batch', type=int, default=64, help='Размер батча.')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения.')
    parser.add_argument('--project', type=str, default='/app/data/05_runs', help='Папка для сохранения результатов.')
    parser.add_argument('--name', type=str, default='experiment', help='Имя для папки запуска.')

    # Параметры аугментации
    parser.add_argument('--augment', action='store_true', help='Включить аугментацию.')
    parser.add_argument('--degrees', type=float, default=5.0, help='Градус случайных поворотов.')
    parser.add_argument('--translate', type=float, default=0.4, help='Доля случайных сдвигов.')
    parser.add_argument('--scale', type=float, default=0.5, help='Коэффициент случайного масштабирования.')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Вероятность отражения по горизонтали.')
    parser.add_argument('--hsv_h', type=float, default=0.4, help='Изменение тона (Hue).')
    parser.add_argument('--hsv_s', type=float, default=0.4, help='Изменение насыщенности (Saturation).')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='Изменение яркости (Value).')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Вероятность применения мозаичной аугментации (0.0 - выключить).')
    parser.add_argument('--close_mosaic', type=int, default=20, help='Количество последних эпох, на которых нужно ОТКЛЮЧИТЬ мозаику.')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='Вероятность применения аугментации Copy-Paste.')
    parser.add_argument('--shear', type=float, default=0.0, help='Градус сдвига (shear).')
    parser.add_argument('--perspective', type=float, default=0.0, help='Коэффициент перспективных искажений.')
    
    # Параметры оптимизатора
    parser.add_argument('--optimizer', type=str, default='auto', help='Оптимизатор (SGD, Adam, etc.).')
    parser.add_argument('--cos_lr', action='store_true', help='Использовать косинусный планировщик скорости обучения.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Коэффициент затухания весов (регуляризация).')
    parser.add_argument('--mixup', type=float, default=0.0, help='Вероятность применения аугментации MixUp.')

    args = parser.parse_args()
    
    # --- НАСТРОЙКА MLFLOW ---
    # Устанавливаем URI для подключения к серверу MLflow в Docker
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    # Устанавливаем (или создаем) эксперимент
    mlflow.set_experiment(args.mlflow_experiment_name)

    # --- ЗАПУСК ОБУЧЕНИЯ С ЛОГИРОВАНИЕМ ---
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(vars(args))
        
        # Загружаем модель
        model = YOLO(args.model)
        
        # Запускаем обучение
        # YOLOv8 автоматически логирует метрики в MLflow, если он активен!
        model.train(
            data=args.config,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            augment=args.augment,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            fliplr=args.fliplr,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            mosaic=args.mosaic,
            copy_paste=args.copy_paste,
            close_mosaic=args.close_mosaic,
            shear=args.shear,
            perspective=args.perspective,
            optimizer=args.optimizer,
            cos_lr=args.cos_lr,
            weight_decay=args.weight_decay,
            mixup=args.mixup
        )
        
        # --- ЛОГИРОВАНИЕ АРТЕФАКТОВ ---
        print("Логирование артефактов в MLflow...")
        
        # 1. Логируем конфигурационный файл датасета
        config_path = Path(args.config)
        if config_path.exists():
            mlflow.log_artifact(str(config_path), "configs")
        
        # 2. Находим путь к папке с результатами этого запуска
        results_path = Path(args.project) / args.name
        
        # 3. Логируем лучшие веса
        best_weights_path = results_path / "weights" / "best.pt"
        if best_weights_path.exists():
            mlflow.log_artifact(str(best_weights_path), "models")
            
        # 4. Логируем все графики и CSV
        for artifact_file in results_path.glob('*.*'):
            if artifact_file.suffix in ['.png', '.jpg', '.csv']:
                mlflow.log_artifact(str(artifact_file), "results_visuals")

    print("\n--- Процесс Завершен! ---")
    print(f"Результаты эксперимента залогированы в MLflow. Эксперимент: '{args.mlflow_experiment_name}'")
