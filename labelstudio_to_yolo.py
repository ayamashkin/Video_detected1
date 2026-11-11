#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный конвертер Label Studio → Ultralytics YOLOv8.
Рекурсивно ищет изображения и .txt во ВСЕХ подпапках.

Использование:
    python labelstudio_to_yolo.py --input raw_labelstudio --output dataset_labelstudio --split 0.8
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Label Studio → Ultralytics YOLOv8")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Папка с датасетом (рекурсивный поиск)")
    parser.add_argument("--output", "-o", type=str, default="dataset_labelstudio",
                        help="Выходная папка")
    parser.add_argument("--split", "-s", type=float, default=0.8,
                        help="Доля train")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def read_classes(classes_file):
    classes = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    return classes


def recursive_find_files(root, extensions):
    """Рекурсивно ищет файлы с указанными расширениями."""
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
        # Для Windows — ищем и заглавные расширения
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))  # убираем дубликаты


def find_txt_for_image(img_path, all_txt_files):
    """Ищет .txt для изображения."""
    stem = img_path.stem

    # Exact match в той же папке
    exact = img_path.with_suffix(".txt")
    if exact.exists():
        return exact

    # Fuzzy: stem содержится в имени .txt
    for txt in all_txt_files:
        if stem in txt.name:
            return txt

    # Обратный fuzzy: имя .txt (без префикса) = stem изображения
    for txt in all_txt_files:
        # Убираем возможный префикс (например, "0a1eec7a-IMG...")
        txt_stem = txt.stem
        if "-" in txt_stem:
            parts = txt_stem.split("-")
            for part in parts:
                if part == stem:
                    return txt
        if stem == txt_stem:
            return txt

    return None


def main():
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)

    if not src.exists():
        print(f"[ERROR] Папка не найдена: {src}")
        return

    print(f"[INFO] Источник: {src.resolve()}")
    print(f"[INFO] Выход: {out.resolve()}")

    # ── Читаем классы ──
    classes_file = src / "classes.txt"
    notes_file = src / "notes.json"

    classes = None
    if classes_file.exists():
        classes = read_classes(classes_file)
        print(f"[INFO] Классы из classes.txt: {classes}")
    elif notes_file.exists():
        with open(notes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = data.get("categories", [])
        categories.sort(key=lambda x: x["id"])
        classes = [c["name"] for c in categories]
        print(f"[INFO] Классы из notes.json: {classes}")
    else:
        # Ищем classes.txt рекурсивно
        found = list(src.rglob("classes.txt"))
        if found:
            classes_file = found[0]
            classes = read_classes(classes_file)
            print(f"[INFO] Классы из {classes_file}: {classes}")

    if not classes:
        print("[ERROR] Не найден classes.txt или notes.json")
        return

    # ── Рекурсивный поиск ВСЕХ изображений и .txt ──
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    images = recursive_find_files(src, image_exts)

    # Исключаем системные .txt
    all_txt_raw = recursive_find_files(src, [".txt"])
    all_txt_files = [t for t in all_txt_raw
                     if t.name.lower() not in {"classes.txt", "train.txt", "val.txt", "test.txt", "notes.txt"}]

    print(f"[INFO] Найдено {len(images)} изображений (рекурсивно)")
    print(f"[INFO] Найдено {len(all_txt_files)} .txt файлов разметки (рекурсивно)")

    # Диагностика — покажем первые 5 файлов
    if images:
        print(f"[DEBUG] Первые изображения:")
        for img in images[:5]:
            print(f"  {img.relative_to(src)}")
    if all_txt_files:
        print(f"[DEBUG] Первые .txt:")
        for txt in all_txt_files[:5]:
            print(f"  {txt.relative_to(src)}")

    # ── Ищем пары ──
    pairs = []
    unmatched_images = []
    for img in images:
        lbl = find_txt_for_image(img, all_txt_files)
        if lbl:
            pairs.append((img, lbl))
        else:
            unmatched_images.append(str(img.relative_to(src)))

    if unmatched_images:
        print(f"[WARN] {len(unmatched_images)} изображений без разметки:")
        for name in unmatched_images[:5]:
            print(f"  - {name}")
        if len(unmatched_images) > 5:
            print(f"  ... и ещё {len(unmatched_images)-5}")

    if not pairs:
        print("[ERROR] Не найдено пар изображение + .txt")
        print("[HINT] Проверьте, что .txt файлы содержат имя .jpg в названии")
        return

    print(f"[INFO] Сопоставлено {len(pairs)} пар изображение+разметка")

    # ── Разделение train/val ──
    import random
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    if not val_pairs and train_pairs:
        print("[WARN] Val пустой, берём 10%% из train")
        val_count = max(1, len(train_pairs) // 10)
        val_pairs = train_pairs[-val_count:]
        train_pairs = train_pairs[:-val_count]

    # ── Копирование ──
    stats = Counter()

    def copy_subset(subset_pairs, subset_name):
        img_out = out / "images" / subset_name
        lbl_out = out / "labels" / subset_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img, lbl in subset_pairs:
            shutil.copy2(img, img_out / img.name)
            dst_lbl = lbl_out / (img.stem + ".txt")
            shutil.copy2(lbl, dst_lbl)
            with open(lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        cls_name = classes[cls_id] if cls_id < len(classes) else f"unknown_{cls_id}"
                        stats[cls_name] += 1
        print(f"  [OK] {subset_name}: {len(subset_pairs)} изображений")

    copy_subset(train_pairs, "train")
    if val_pairs:
        copy_subset(val_pairs, "val")

    # ── data.yaml ──
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"[OK] Конвертация завершена!")
    print(f"[OK] data.yaml: {out / 'data.yaml'}")
    print(f"[OK] Классы ({len(classes)}):")
    for i, name in enumerate(classes):
        cnt = stats.get(name, 0)
        print(f"     {i}: {name:<15} — {cnt} объектов")
    print(f"{'='*60}")
    print(f"\nОбучение на GPU:")
    print(f"  python train_gpu.py --data {out}/data.yaml --epochs 100 --imgsz 640 --batch 16")
    print(f"\nОбучение на CPU:")
    print(f"  python train_cpu.py --data {out}/data.yaml --epochs 50 --imgsz 320 --batch 4")


if __name__ == "__main__":
    main()