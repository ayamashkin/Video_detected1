#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединение нескольких YOLO-датасетов в один мультиклассовый.

Поддерживает:
  - Ultralytics YOLOv8 (dataset/data.yaml)
  - CWD1500 (dataset/data.yaml с obj.names)
  - Label Studio (dataset/data.yaml с classes.txt)

Использование:
    # Объединить CWD1500 + Label Studio
    python merge_datasets.py \
        --datasets \
            "cwd:dataset_cwd1500/data.yaml:wheel,car" \
            "ls:dataset_labelstudio/data.yaml:car,manometer,person,wheel" \
        --output dataset_merged

Каждый --datasets аргумент: "имя:путь_к_data.yaml:классы_через_запятую"
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Объединение YOLO-датасетов")
    parser.add_argument("--datasets", "-d", type=str, nargs="+", required=True,
                        help="Список датасетов: name:path/to/data.yaml:classes")
    parser.add_argument("--output", "-o", type=str, default="dataset_merged",
                        help="Выходная папка")
    parser.add_argument("--split", type=float, default=None,
                        help="Пере-разделить train/val (0.8). Если None — сохраняем исходное")
    return parser.parse_args()


def parse_dataset_spec(spec):
    """Парсит 'name:path:cls1,cls2'."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Неверный формат: {spec}. Ожидается: name:path:data.yaml:classes")
    name = parts[0]
    yaml_path = Path(parts[1])
    classes = [c.strip() for c in parts[2].split(",")]
    return name, yaml_path, classes


def read_data_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def copy_with_remap(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir,
                    class_map, global_stats):
    """Копирует изображения и перенумеровывает class_id."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_file in sorted(src_img_dir.iterdir()):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl_file = src_lbl_dir / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue

        new_lines = []
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                if old_cls not in class_map:
                    continue
                new_cls = class_map[old_cls]
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
                global_stats[new_cls] += 1

        if not new_lines:
            continue

        shutil.copy2(img_file, dst_img_dir / img_file.name)
        with open(dst_lbl_dir / (img_file.stem + ".txt"), "w") as f:
            f.writelines(new_lines)
        copied += 1

    return copied


def main():
    args = parse_args()
    out_dir = Path(args.output)

    # Собираем глобальный список классов
    global_classes = []
    class_sources = []

    for spec in args.datasets:
        name, yaml_path, wanted_classes = parse_dataset_spec(spec)
        if not yaml_path.exists():
            print(f"[ERROR] Не найден: {yaml_path}")
            continue
        data = read_data_yaml(yaml_path)
        local_names = data.get("names", {})
        if isinstance(local_names, dict):
            local_names = {int(k): v for k, v in local_names.items()}
        else:
            local_names = {i: v for i, v in enumerate(local_names)}

        print(f"[INFO] Датасет '{name}': {yaml_path}")
        print(f"       Локальные классы: {local_names}")
        print(f"       Нужные классы: {wanted_classes}")
        class_sources.append((name, yaml_path, wanted_classes, data, local_names))

    # Строим глобальную нумерацию
    for name, yaml_path, wanted_classes, data, local_names in class_sources:
        for local_id, local_name in local_names.items():
            if local_name in wanted_classes and local_name not in global_classes:
                global_classes.append(local_name)

    print(f"\n[INFO] Итоговые классы ({len(global_classes)}):")
    for i, cls in enumerate(global_classes):
        print(f"  {i}: {cls}")

    # Копируем и перенумеровываем
    stats = Counter()
    total_copied = 0

    for name, yaml_path, wanted_classes, data, local_names in class_sources:
        class_map = {}
        for local_id, local_name in local_names.items():
            if local_name in wanted_classes and local_name in global_classes:
                class_map[local_id] = global_classes.index(local_name)

        if not class_map:
            print(f"[WARN] Нет совпадающих классов для '{name}'")
            continue

        src_root = yaml_path.parent
        for subset in ["train", "val", "test"]:
            src_img = src_root / "images" / subset
            src_lbl = src_root / "labels" / subset
            if not src_img.exists() or not src_lbl.exists():
                continue

            dst_img = out_dir / "images" / subset
            dst_lbl = out_dir / "labels" / subset
            copied = copy_with_remap(src_img, src_lbl, dst_img, dst_lbl, class_map, stats)
            total_copied += copied
            print(f"  [OK] {name}/{subset}: {copied} изображений")

    # data.yaml
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(global_classes)},
        "nc": len(global_classes),
    }
    if (out_dir / "images" / "test").exists():
        data_yaml["test"] = "images/test"

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"[OK] Объединение завершено!")
    print(f"[OK] Всего изображений: {total_copied}")
    print(f"[OK] Классы: {global_classes}")
    print(f"[OK] data.yaml: {out_dir / 'data.yaml'}")
    print(f"{'='*60}")
    print(f"\nОбучение на GPU:")
    print(f"  python train_gpu.py --data {out_dir}/data.yaml --epochs 100 --imgsz 640 --batch 16")
    print(f"\nОбучение на CPU:")
    print(f"  python train_cpu.py --data {out_dir}/data.yaml --epochs 50 --imgsz 320 --batch 4")


if __name__ == "__main__":
    main()