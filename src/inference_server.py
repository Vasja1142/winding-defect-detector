# src/inference_server.py

import cv2
import argparse
from ultralytics import YOLO
from flask import Flask, Response

# --- Глобальные переменные ---
app = Flask(__name__)
model = None
source = None
frame_skip = 1
confidence_threshold = 0.5
imgsz = 640  # <-- Новая глобальная переменная

def generate_frames():
    """Генератор, который захватывает кадры, обрабатывает их и отдает как jpeg."""
    # Используем try-except, чтобы элегантно обработать ошибку, если источник - число
    try:
        source_int = int(source)
        cap = cv2.VideoCapture(source_int)
    except ValueError:
        cap = cv2.VideoCapture(source)
        
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть источник видео: {source}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        results = model(frame, imgsz=imgsz, conf=confidence_threshold, verbose=False) # <-- Используем imgsz
        annotated_frame = results[0].plot()

        if len(results[0].boxes) > 0:
            cv2.putText(annotated_frame, "DEFECT DETECTED!", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Стриминговый сервер для детекции.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--frame_skip', type=int, default=1) # <-- Поставим 1, чтобы обрабатывать каждый кадр
    parser.add_argument('--confidence', type=float, default=0.6)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения для инференса.') # <-- Новый аргумент
    args = parser.parse_args()

    # Инициализируем глобальные переменные
    model = YOLO(args.model_path)
    source = args.source
    frame_skip = args.frame_skip
    confidence_threshold = args.confidence
    imgsz = args.imgsz # <-- Сохраняем новый аргумент

    print(f"[*] Запуск сервера на http://{args.host}:{args.port}")
    print(f"[*] Стрим доступен по адресу: http://<ВАШ_IP_АДРЕС>:{args.port}/video_feed")
    
    app.run(host=args.host, port=args.port, debug=False)