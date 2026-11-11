#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт дообучения YOLOv8 под кастомные классы.

Использование:
    python train_custom.py --data data.yaml --epochs 100 --imgsz 640 --batch 16

Требования:
    pip install ultralytics

Структура датасета (см. dataset_prep.py):
    dataset/
      images/
        train/  *.jpg
        val/    *.jpg
      labels/
        train/  *.txt  (YOLO format: class x_center y_center width height)
        val/    *.txt
      data.yaml
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 для кастомных объектов")
    parser.add_argument("--data", "-d", type=str, default="dataset/data.yaml",
                        help="Путь к data.yaml (конфиг датасета)")
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt",
                        help="Базовая модель: yolov8n.pt, yolov8s.pt, yolov8m.pt")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="Количество эпох")
    parser.add_argument("--imgsz", "-i", type=int, default=640,
                        help="Размер входа")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="Размер батча")
    parser.add_argument("--device", type=str, default="0",
                        help="Устройство: 0=GPU, cpu=CPU, mps=Apple Silicon")
    parser.add_argument("--project", "-p", type=str, default="runs/detect",
                        help="Папка для сохранения результатов")
    parser.add_argument("--name", "-n", type=str, default="custom_train",
                        help="Имя эксперимента")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (эпох без улучшения)")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить обучение с чекпоинта")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Заморозить первые N слоёв (transfer learning)")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml не найден: {data_path}")
        print("Сначала запустите: python dataset_prep.py")
        sys.exit(1)

    print(f"[INFO] Базовая модель: {args.model}")
    print(f"[INFO] Датасет: {args.data}")
    print(f"[INFO] Эпох: {args.epochs}, batch: {args.batch}, imgsz: {args.imgsz}")
    print(f"[INFO] Устройство: {args.device}")
    print(f"[INFO] Заморозка первых {args.freeze} слоёв")

    # Загрузка базовой модели
    model = YOLO(args.model)

    # Обучение
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        freeze=args.freeze,
        # Оптимизации
        amp=True,           # Автоматический mixed precision
        cos_lr=True,        # Cosine learning rate schedule
        close_mosaic=10,    # Отключить mosaic на последних 10 эпохах
        # Валидация
        val=True,
        save=True,
        plots=True,
    )

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"[OK] Обучение завершено!")
    print(f"[OK] Лучший вес: {best_path}")
    print(f"[OK] mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"[OK] mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"{'='*60}")
    print(f"\nДля использования в проекте:")
    print(f"  1. Скопируйте {best_path} → models/best.pt")
    print(f"  2. В config.yaml установите: model: 'models/best.pt'")
    print(f"  3. В config.yaml установите: classes.mode: 'coco' (или custom)")


if __name__ == "__main__":
    main()