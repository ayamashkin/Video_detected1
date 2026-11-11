#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированное обучение YOLOv8 на CPU.
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 на CPU")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Путь к data.yaml")
    parser.add_argument("--model", "-m", type=str, default="models/yolov8n.pt",
                        help="Базовая модель")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Количество эпох")
    parser.add_argument("--imgsz", "-i", type=int, default=320,
                        help="Размер входа")
    parser.add_argument("--batch", "-b", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--freeze", "-f", type=int, default=0,
                        help="Заморозить первых N слоёв")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--project", "-p", type=str, default=None,
                        help="Папка для результатов (по умолчанию: ./runs/detect)")
    parser.add_argument("--name", "-n", type=str, default="cwd1500_cpu",
                        help="Имя эксперимента")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить обучение")
    parser.add_argument("--optimizer", type=str, default="SGD",
                        choices=["SGD", "Adam", "AdamW", "RMSProp"],
                        help="Оптимизатор")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml не найден: {data_path}")
        return

    # FIX: абсолютный путь для project, чтобы не было путаницы с дисками/папками
    if args.project:
        project_path = Path(args.project).resolve()
    else:
        project_path = Path.cwd() / "runs" / "detect"
    project_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"[INFO] Обучение YOLOv8 на CPU")
    print(f"[INFO] Модель: {args.model}")
    print(f"[INFO] Датасет: {args.data}")
    print(f"[INFO] Эпох: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")
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

    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        freeze=args.freeze,
        optimizer=args.optimizer,
        amp=False,
        cos_lr=True,
        close_mosaic=5,
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
        mixup=0.0,
        copy_paste=0.0,
        val=True,
        save=True,
        plots=True,
    )

    elapsed = time.time() - start_time
    best_path = project_path / args.name / "weights" / "best.pt"
    last_path = project_path / args.name / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"[OK] Обучение завершено за {elapsed/3600:.1f} часов")
    print(f"[OK] Лучший вес: {best_path}")
    print(f"[OK] mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"[OK] mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"{'='*60}")
    print(f"\nДля использования в проекте:")
    print(f"  copy {best_path} models\best.pt")
    print(f"\nПродолжить обучение:")
    print(f"  python train_cpu.py --data {args.data} --epochs {args.epochs} --imgsz {args.imgsz} --resume")


if __name__ == "__main__":
    main()