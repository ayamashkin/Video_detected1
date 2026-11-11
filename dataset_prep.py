#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подготовка датасета YOLO для дообучения.

Использование:
    # 1. Положите фото в папку raw_images/
    # 2. Разметите через LabelImg / CVAT / Roboflow (YOLO format)
    # 3. Запустите:
    python dataset_prep.py --raw raw_images --labels raw_labels --split 0.8

Автоматическая загрузка из Roboflow (опционально):
    python dataset_prep.py --roboflow workspace/project/version --api-key XXX
"""

import argparse
import os
import shutil
import random
import zipfile
from pathlib import Path
from collections import Counter

import yaml


CLASSES = [
    "wheel",
    "pressure gauge",
    "monitor",
    "laptop",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Подготовка YOLO-датасета")
    parser.add_argument("--raw", "-r", type=str, default="raw_images",
                        help="Папка с исходными изображениями")
    parser.add_argument("--labels", "-l", type=str, default="raw_labels",
                        help="Папка с YOLO-разметкой (.txt)")
    parser.add_argument("--output", "-o", type=str, default="dataset",
                        help="Выходная папка")
    parser.add_argument("--split", "-s", type=float, default=0.8,
                        help="Доля train (0.8 = 80% train, 20% val)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--roboflow", type=str, default=None,
                        help="Roboflow проект: workspace/project/version")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key")
    return parser.parse_args()


def download_from_roboflow(project, api_key, output_dir):
    """Скачать датасет из Roboflow."""
    try:
        import roboflow
    except ImportError:
        print("[INFO] Установка roboflow...")
        os.system("pip install roboflow")
        import roboflow

    rf = roboflow.Roboflow(api_key=api_key)
    workspace, project_name, version = project.split("/")
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(int(version)).download("yolov8", location=output_dir)
    print(f"[OK] Скачан из Roboflow: {output_dir}")
    return output_dir


def prepare_dataset(raw_images, raw_labels, output_dir, split_ratio, seed):
    """Разделить на train/val и создать data.yaml."""
    random.seed(seed)
    output_dir = Path(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    raw_images = Path(raw_images)
    raw_labels = Path(raw_labels)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([f for f in raw_images.iterdir() if f.suffix.lower() in image_exts])

    if not images:
        print(f"[ERROR] Нет изображений в {raw_images}")
        return False

    # Проверка наличия labels
    missing_labels = []
    for img in images:
        label_file = raw_labels / (img.stem + ".txt")
        if not label_file.exists():
            missing_labels.append(img.name)

    if missing_labels:
        print(f"[WARN] {len(missing_labels)} изображений без разметки:")
        for m in missing_labels[:5]:
            print(f"  - {m}")
        if len(missing_labels) > 5:
            print(f"  ... и ещё {len(missing_labels)-5}")
        # Фильтруем только с разметкой
        images = [img for img in images if (raw_labels / (img.stem + ".txt")).exists()]

    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Копирование
    stats = Counter()
    for subset, img_list in [("train", train_images), ("val", val_images)]:
        for img in img_list:
            shutil.copy2(img, output_dir / "images" / subset / img.name)
            label_src = raw_labels / (img.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, output_dir / "labels" / subset / (img.stem + ".txt"))
                # Подсчёт классов
                with open(label_src) as f:
                    for line in f:
                        cls_id = int(line.strip().split()[0])
                        stats[f"{subset}_{cls_id}"] += 1

    # Создание data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
        "nc": len(CLASSES),
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"[OK] Датасет готов: {output_dir}")
    print(f"[OK] Train: {len(train_images)} | Val: {len(val_images)}")
    print(f"[OK] Классов: {len(CLASSES)}")
    for i, name in enumerate(CLASSES):
        train_cnt = stats.get(f"train_{i}", 0)
        val_cnt = stats.get(f"val_{i}", 0)
        print(f"  {i}: {name:<18} train={train_cnt:>4}  val={val_cnt:>4}")
    print(f"{'='*60}")
    print(f"\nСледующий шаг:")
    print(f"  python train_custom.py --data {output_dir}/data.yaml")
    return True


def main():
    args = parse_args()

    if args.roboflow and args.api_key:
        print("[INFO] Загрузка из Roboflow...")
        download_from_roboflow(args.roboflow, args.api_key, args.output)
        return

    if not Path(args.raw).exists():
        print(f"[ERROR] Папка не найдена: {args.raw}")
        print("Создайте структуру:")
        print("  raw_images/  — фотографии")
        print("  raw_labels/  — .txt файлы YOLO-формата")
        print("\nИли используйте --roboflow для загрузки из Roboflow")
        return

    prepare_dataset(args.raw, args.labels, args.output, args.split, args.seed)


if __name__ == "__main__":
    main()