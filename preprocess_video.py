#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Экспорт YOLO-моделей в оптимизированные форматы для ускорения инференса.

Поддерживаемые форматы:
    - ONNX (универсальный, работает на CPU/GPU)
    - TensorRT .engine (NVIDIA GPU, максимальная скорость)
    - OpenVINO (Intel CPU/GPU)

Использование:
    python export_model.py --model models/yolov8s-worldv2.pt --format engine --imgsz 480
    python export_model.py --model models/yolov8n.pt --format onnx --imgsz 320
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Экспорт YOLO в оптимизированные форматы")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Путь к .pt модели")
    parser.add_argument("--format", "-f", type=str, default="onnx",
                        choices=["onnx", "engine", "openvino", "torchscript"],
                        help="Целевой формат")
    parser.add_argument("--imgsz", "-i", type=int, default=640,
                        help="Размер входа")
    parser.add_argument("--half", action="store_true",
                        help="FP16 (half precision) — только для GPU")
    parser.add_argument("--device", "-d", type=str, default="0",
                        help="Устройство для экспорта (0=GPU, cpu=CPU)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic batch size (для ONNX/TensorRT)")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplify ONNX graph")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERROR] Модель не найдена: {model_path}")
        sys.exit(1)

    print(f"[INFO] Загрузка модели: {model_path}")
    model = YOLO(str(model_path))

    print(f"[INFO] Экспорт в {args.format.upper()}, imgsz={args.imgsz}, half={args.half}")

    try:
        model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            device=args.device,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )

        # Определяем выходной файл
        ext_map = {
            "onnx": ".onnx",
            "engine": ".engine",
            "openvino": "_openvino_model",
            "torchscript": ".torchscript",
        }

        if args.format == "openvino":
            out_path = model_path.with_suffix("").name + "_openvino_model"
        else:
            out_path = model_path.with_suffix(ext_map[args.format])

        print(f"\n{'='*60}")
        print(f"[OK] Экспорт завершён!")
        print(f"[OK] Файл: {out_path}")
        print(f"[OK] Используйте в config.yaml:")
        print(f"     model: '{out_path}'")
        print(f"{'='*60}")

    except Exception as e:
        print(f"[ERROR] Экспорт не удался: {e}")
        if args.format == "engine":
            print("[HINT] Для TensorRT нужен NVIDIA GPU + CUDA + TensorRT")
            print("[HINT] Установите: pip install tensorrt")
        sys.exit(1)


if __name__ == "__main__":
    main()