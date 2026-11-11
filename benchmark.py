#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Бенчмарк инференса YOLO. Подбирает оптимальные параметры для вашего железа.

Использование:
    python benchmark.py --video data/video.mp4 --model models/yolov8s-worldv2.pt
    python benchmark.py --video data/video.mp4 --model models/yolov8s-worldv2.pt --imgsz 480 --device 0 --half
    python benchmark.py --video data/video.mp4 --model models/yolov8n.pt --device cpu --skip-frames 2
"""

import argparse
import time
import cv2
from pathlib import Path
from ultralytics import YOLO


def benchmark(model, cap, device, imgsz, half, skip_frames, max_frames=300):
    """Замер FPS для заданной конфигурации."""
    frame_count = 0
    infer_count = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        _ = model(frame, verbose=False, device=device, imgsz=imgsz, half=half)
        infer_count += 1

        if infer_count >= max_frames:
            break

    elapsed = time.time() - start
    fps = infer_count / elapsed if elapsed > 0 else 0
    return fps, infer_count, elapsed


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 для кастомных объектов")
    parser.add_argument("--video", "-v", type=str, required=True,
                        help="Путь к видеофайлу")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Путь к модели (.pt, .engine, .onnx)")
    parser.add_argument("--device", "-d", type=str, default="0",
                        help="Устройство: 0=GPU, cpu=CPU, mps=Apple Silicon")
    parser.add_argument("--imgsz", "-i", type=int, default=640,
                        help="Размер входа (320, 480, 640)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 half precision (только GPU)")
    parser.add_argument("--skip-frames", "-s", type=int, default=1,
                        help="Пропуск кадров: 1=все, 2=каждый 2-й")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Макс. кадров для теста")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results.txt",
                        help="Файл для сохранения результатов")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Видео: {args.video}")
    print(f"[INFO] Кадров: {total_frames}, FPS видео: {video_fps:.1f}")
    print(f"[INFO] Модель: {args.model}")
    print(f"[INFO] Устройство: {args.device}")
    print("=" * 60)

    model = YOLO(args.model)

    # Если заданы конкретные параметры — тестируем только их
    if args.imgsz != 640 or args.half or args.skip_frames != 1:
        print(f"[INFO] Тест с заданными параметрами:")
        print(f"       imgsz={args.imgsz}, half={args.half}, skip={args.skip_frames}")

        fps, count, elapsed = benchmark(
            model, cap, args.device, args.imgsz, args.half, args.skip_frames, args.max_frames
        )
        print(f"[OK] Результат: {fps:.1f} FPS ({count} кадров за {elapsed:.1f}с)")

        # Сравнение с FPS видео
        if video_fps > 0:
            ratio = fps / video_fps
            status = "✅ Быстрее видео" if ratio >= 1 else "⚠️ Медленнее видео"
            print(f"     {status}: инференс {fps:.1f} FPS / видео {video_fps:.1f} FPS = {ratio:.2f}x")

        cap.release()
        return

    # Иначе — полный бенчмарк с перебором конфигураций
    configs = [
        (640, False, 1),
        (480, False, 1),
        (320, False, 1),
        (640, True, 1),
        (480, True, 1),
        (320, True, 1),
        (640, False, 2),
        (480, False, 2),
        (320, False, 2),
    ]

    results = []
    for imgsz, half, skip in configs:
        if args.device == "cpu" and half:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        label = f"imgsz={imgsz}, half={half}, skip={skip}"
        print(f"  Тест: {label} ...", end=" ")

        try:
            fps, count, elapsed = benchmark(model, cap, args.device, imgsz, half, skip, args.max_frames)
            print(f"{fps:.1f} FPS ({count} кадров за {elapsed:.1f}с)")
            results.append((label, fps, count, elapsed))
        except Exception as e:
            print(f"ERROR: {e}")

    cap.release()

    results.sort(key=lambda x: -x[1])

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ (от лучшего к худшему)")
    print("=" * 60)
    for label, fps, count, elapsed in results:
        marker = " ★ BEST" if fps == results[0][1] else ""
        print(f"  {label:<35} → {fps:>6.1f} FPS{marker}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Video FPS: {video_fps:.1f}\n\n")
        for label, fps, count, elapsed in results:
            f.write(f"{label}: {fps:.1f} FPS\n")
    print(f"\n[OK] Результаты сохранены: {args.output}")


if __name__ == "__main__":
    main()