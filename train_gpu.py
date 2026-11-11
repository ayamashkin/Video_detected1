#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированное обучение YOLOv8 на GPU (CUDA).

Рекомендуемые параметры для разных GPU:
    - RTX 3060 (12 GB): batch=16, imgsz=640
    - RTX 4060 (8 GB):  batch=8, imgsz=640
    - GTX 1660 (6 GB):  batch=4, imgsz=640
    - Colab T4 (16 GB): batch=16, imgsz=640

Использование:
    # Полное обучение (100 эпох, imgsz=640)
    python train_gpu.py --data dataset_labelstudio/data.yaml --epochs 100 --imgsz 640 --batch 16

    # Быстрая проверка (20 эпох)
    python train_gpu.py --data dataset_labelstudio/data.yaml --epochs 20 --imgsz 640 --batch 16

    # Продолжить прерванное
    python train_gpu.py --data dataset_labelstudio/data.yaml --epochs 100 --imgsz 640 --batch 16 --resume

    # С заморозкой backbone (transfer learning, быстрее)
    python train_gpu.py --data dataset_labelstudio/data.yaml --epochs 100 --imgsz 640 --batch 16 --freeze 10
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 на GPU")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Путь к data.yaml")
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt",
                        help="Базовая модель (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="Количество эпох")
    parser.add_argument("--imgsz", "-i", type=int, default=640,
                        help="Размер входа: 320/416/640")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="Batch size (4-16 в зависимости от VRAM)")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="DataLoader workers (4-8 для GPU)")
    parser.add_argument("--freeze", "-f", type=int, default=0,
                        help="Заморозить первых N слоёв (10-15 для transfer learning)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--project", "-p", type=str, default=None,
                        help="Папка для результатов (по умолчанию: ./runs/detect)")
    parser.add_argument("--name", "-n", type=str, default="custom_gpu",
                        help="Имя эксперимента")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить обучение с последнего чекпоинта")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW", "RMSProp"],
                        help="Оптимизатор (AdamW рекомендуется для GPU)")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device: 0=GPU #0, 0,1=multi-GPU, cpu=CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml не найден: {data_path}")
        return

    # Абсолютный путь для project
    if args.project:
        project_path = Path(args.project).resolve()
    else:
        project_path = Path.cwd() / "runs" / "detect"
    project_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"[INFO] Обучение YOLOv8 на GPU")
    print(f"[INFO] Модель: {args.model}")
    print(f"[INFO] Датасет: {args.data}")
    print(f"[INFO] Эпох: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")
    print(f"[INFO] Device: cuda:{args.device}")
    print(f"[INFO] Optimizer: {args.optimizer}, Workers: {args.workers}")
    print(f"[INFO] Project: {project_path}")
    print(f"[INFO] Name: {args.name}")
    print(f"[INFO] Заморозка слоёв: {args.freeze}")
    print(f"{'='*60}\n")

    start_time = time.time()

    if args.resume:
        last_pt = project_path / args.name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"[INFO] Продолжаем с: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"[WARN] Чекпоинт не найден, начинаем с нуля: {args.model}")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    # GPU-оптимизации
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        freeze=args.freeze,
        optimizer=args.optimizer,
        # GPU-оптимизации
        amp=True,               # Mixed precision (FP16) — +30% скорости на GPU
        cos_lr=True,            # Cosine learning rate
        close_mosaic=10,        # Отключить mosaic на последних 10 эпохах
        # Аугментации (полный набор для GPU)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Валидация
        val=True,
        save=True,
        plots=True,
    )

    elapsed = time.time() - start_time
    best_path = project_path / args.name / "weights" / "best.pt"
    last_path = project_path / args.name / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"[OK] Обучение завершено за {elapsed/60:.1f} минут")
    print(f"[OK] Лучший вес: {best_path}")
    print(f"[OK] mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"[OK] mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"{'='*60}")
    print(f"\nДля использования в проекте:")
    print(f"  copy {best_path} models\best.pt")
    print(f"\nПродолжить обучение:")
    print(f"  python train_gpu.py --data {args.data} --epochs {args.epochs} --imgsz {args.imgsz} --batch {args.batch} --resume")


if __name__ == "__main__":
    main()