#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертация датасета CWD1500 → Ultralytics YOLOv8.

Поддерживает структуры:
  A) train/CWD1500/CWD1500_train/obj_train_data/  (ваш случай)
  B) data/obj_train_data/ + obj.names + train.txt
  C) data/images/train/ + data/labels/train/

Использование:
    python convert_cwd1500.py --input train/CWD1500 --output dataset_cwd1500

После конвертации:
    python train_cpu.py --data dataset_cwd1500/data.yaml --epochs 50 --imgsz 320
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Конвертация CWD1500 → Ultralytics YOLOv8")
    parser.add_argument("--input", "-i", type=str, default="train/CWD1500",
                        help="Корневая папка датасета (где лежат CWD1500_train/CWD1500_val)")
    parser.add_argument("--output", "-o", type=str, default="dataset_cwd1500",
                        help="Выходная папка")
    return parser.parse_args()


def find_subsets(src_dir):
    """
    Авто-определение структуры датасета.
    Возвращает dict: {subset_name: (img_dir, lbl_dir, names_file)}
    """
    subsets = {}

    # Структура A: CWD1500_train/obj_train_data/, CWD1500_val/obj_train_data/
    for subset in ["train", "val", "test"]:
        for prefix in ["CWD1500_", "", "obj_"]:
            candidate = src_dir / f"{prefix}{subset}"
            if candidate.exists():
                img_dir = candidate / "obj_train_data"
                lbl_dir = img_dir  # .txt рядом с .jpg
                names_file = candidate / "obj.names"
                if img_dir.exists():
                    subsets[subset] = (img_dir, lbl_dir, names_file if names_file.exists() else None)
                    break

    # Структура B: единая папка с obj_train_data, obj_val_data
    if not subsets:
        for subset in ["train", "val", "test"]:
            img_dir = src_dir / f"obj_{subset}_data"
            if img_dir.exists():
                lbl_dir = img_dir
                names_file = src_dir / "obj.names"
                subsets[subset] = (img_dir, lbl_dir, names_file if names_file.exists() else None)

    # Структура C: images/train, labels/train
    if not subsets:
        for subset in ["train", "val", "test"]:
            img_dir = src_dir / "images" / subset
            lbl_dir = src_dir / "labels" / subset
            if img_dir.exists() and lbl_dir.exists():
                names_file = src_dir / "obj.names"
                subsets[subset] = (img_dir, lbl_dir, names_file if names_file.exists() else None)

    return subsets


def read_names(names_file):
    """Читает obj.names и возвращает список классов."""
    classes = []
    if not names_file or not names_file.exists():
        return classes
    with open(names_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                classes.append(line)
    return classes


def copy_subset(img_dir, lbl_dir, out_dir, subset_name):
    """Копирует изображения и labels в Ultralytics-структуру."""
    img_out = out_dir / "images" / subset_name
    lbl_out = out_dir / "labels" / subset_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    copied = 0
    missing = 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in image_exts])

    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing += 1
            continue

        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / (img.stem + ".txt"))
        copied += 1

        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    stats[cls_id] += 1

    print(f"  [OK] {subset_name}: {copied} изображений, {missing} без разметки")
    return stats


def main():
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)

    if not src.exists():
        print(f"[ERROR] Папка не найдена: {src}")
        print("Убедитесь, что указали правильный путь, например:")
        print("  python convert_cwd1500.py --input train/CWD1500")
        return

    print(f"[INFO] Источник: {src.resolve()}")
    print(f"[INFO] Выход: {out.resolve()}")

    subsets = find_subsets(src)
    if not subsets:
        print("[ERROR] Не удалось определить структуру датасета")
        print("Ожидается одна из структур:")
        print("  A) CWD1500_train/obj_train_data/  CWD1500_val/obj_train_data/")
        print("  B) obj_train_data/  obj_val_data/")
        print("  C) images/train/  labels/train/")
        return

    print(f"[INFO] Найдены подмножества: {list(subsets.keys())}")

    # Читаем классы из первого найденного names файла
    classes = []
    for subset, (img_dir, lbl_dir, names_file) in subsets.items():
        if names_file and names_file.exists():
            classes = read_names(names_file)
            print(f"[INFO] Классы из {names_file}: {classes}")
            break

    if not classes:
        classes = ["car", "wheel"]
        print(f"[WARN] obj.names не найден, используем дефолт: {classes}")

    # Копируем
    all_stats = Counter()
    for subset, (img_dir, lbl_dir, names_file) in sorted(subsets.items()):
        print(f"\n[INFO] Обработка {subset}...")
        stats = copy_subset(img_dir, lbl_dir, out, subset)
        all_stats.update(stats)

    # data.yaml
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }
    if "test" in subsets:
        data_yaml["test"] = "images/test"

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"[OK] Конвертация завершена!")
    print(f"[OK] data.yaml: {out / 'data.yaml'}")
    print(f"[OK] Классы ({len(classes)}):")
    for i, name in enumerate(classes):
        cnt = all_stats.get(i, 0)
        print(f"     {i}: {name:<15} — {cnt} объектов")
    print(f"{'='*60}")
    print(f"\nСледующий шаг — обучение на CPU:")
    print(f"  python train_cpu.py --data {out}/data.yaml --epochs 20 --imgsz 320")
    print(f"\nЕсли нужно прервать и продолжить позже:")
    print(f"  python train_cpu.py --data {out}/data.yaml --epochs 50 --imgsz 320 --resume")


if __name__ == "__main__":
    main()