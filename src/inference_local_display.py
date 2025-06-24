# src/inference_local_display.py

import cv2
import time
import argparse
from ultralytics import YOLO

def run_local_display(model, source, imgsz, confidence_threshold):
    """
    Захватывает видеопоток, обрабатывает КАЖДЫЙ кадр и отображает результат на локальном экране.
    """
    # Пытаемся преобразовать источник в число (для веб-камер)
    try:
        source = int(source)
    except ValueError:
        pass # Оставляем как есть (для URL или путей к файлам)
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть источник видео: {source}")
        return

    print("Источник видео успешно открыт. Нажмите 'q' для выхода.")
    
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр или видео закончилось. Выход...")
            break

        # --- Обработка КАЖДОГО кадра ---
        # verbose=False отключает текстовый вывод от YOLO для каждого кадра
        results = model(frame, imgsz=imgsz, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # --- Расчет и отображение FPS ---
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Если найдены дефекты - добавляем большой красный текст
        if len(results[0].boxes) > 0:
            cv2.putText(annotated_frame, "DEFECT DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Показываем результат на экране
        cv2.imshow("Real-time Defect Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Работа завершена.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Локальная детекция в реальном времени с отображением.")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к обученной модели (.pt).')
    parser.add_argument('--source', type=str, required=True, help='Источник видео: индекс камеры (0), URL или путь к файлу.')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения для инференса (влияет на скорость).')
    parser.add_argument('--confidence', type=float, default=0.6, help='Порог уверенности.')
    
    args = parser.parse_args()

    model = YOLO(args.model_path)
    run_local_display(model, args.source, args.imgsz, args.confidence)
    