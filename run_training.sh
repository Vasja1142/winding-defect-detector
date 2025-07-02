#!/bin/bash
# Скрипт для запуска обучения модели Winding Defect Detector.

docker compose exec detector python3 src/train.py \
    --model /app/data/00_pretrained/yolo12m.pt \
    --config config.yaml \
    --epochs 250 \
    --batch 16 \
    --project /app/data/05_runs \
    --name WDD_m_v3_e250_b16 \
    --imgsz 640 \
    --patience 50 \
    --augment \
    --copy_paste 0.4 \
    --degrees 2.0 \
    --translate 0.2 \
    --scale 0.5 \
    --shear 1.0 \
    --perspective 0.0004 \
    --fliplr 0.5 \
    --hsv_h 0.85 \
    --hsv_s 1.0 \
    --hsv_v 0.85 \
    --close_mosaic 50 \
    --mosaic 1.0 \
    --mixup 0.3 \
    --optimizer SGD \
    --cos_lr \
    --weight_decay 0.0002 \
    --label_smoothing 0.1 \
    --lr0 0.0002 \
    --lrf 0.04 \
    --cache True

