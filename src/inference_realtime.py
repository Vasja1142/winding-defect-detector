# src/inference_realtime.py

import cv2
import time
import argparse
from ultralytics import YOLO

def run_realtime_inference(model, source, frame_skip, confidence_threshold): # <--- 'source' вместо 'camera_index'
    """
    Захватывает видеопоток с камеры или URL и применяет модель YOLO в реальном времени.
    """
    cap = cv2.VideoCapture(source) # <--- 'source' используется здесь
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть источник видео: {source}")
        return

    print("Источник видео успешно открыт. Нажмите 'q' для выхода.")
    
    frame_count = 0
    last_time = time.time()
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр. Выход...")
            break

        frame_count += 1
        
        if frame_count % frame_skip != 0:
            cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Real-time Defect Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        current_time = time.time()
        if (current_time - last_time) > 0:
             display_fps = frame_skip / (current_time - last_time)
        last_time = current_time

        results = model(frame, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()

        if len(results[0].boxes) > 0:
            print(f"ТРЕВОГА! Обнаружен дефект! Уверенность: {results[0].boxes[0].conf.item():.2f}")
            cv2.putText(annotated_frame, "DEFECT DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.putText(annotated_frame, f"FPS: {display_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-time Defect Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Работа завершена.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для детекции в реальном времени.")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к обученной модели (.pt).')
    
    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ ---
    parser.add_argument('--source', type=str, default='0', help='Источник видео: индекс камеры (0, 1) или URL видеопотока.')
    
    parser.add_argument('--frame_skip', type=int, default=5, help='Анализировать каждый N-ный кадр.')
    parser.add_argument('--confidence', type=float, default=0.6, help='Порог уверенности.')
    
    args = parser.parse_args()

    # --- Умная обработка источника ---
    source = args.source
    try:
        # Пытаемся превратить источник в число (для веб-камер 0, 1, ...)
        source = int(source)
    except ValueError:
        # Если не получилось, значит это строка (URL), оставляем как есть
        pass
        
    model = YOLO(args.model_path)
    
    run_realtime_inference(model, source, args.frame_skip, args.confidence)