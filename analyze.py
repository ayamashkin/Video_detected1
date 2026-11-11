#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
КОНСОЛЬНЫЙ анализатор видеофайла. НЕ содержит Flask / веб-сервер.

Использование:
    python analyze.py --video data/file.mp4 --audit
    python analyze.py --video data/file.mp4 --targets pen,headphones --expected 300 --verbose
"""

import argparse
import time
import cv2
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

from core import load_config, Session, Exporter, SimpleTracker
from interaction_tracker import InteractionTracker


class BatchVideoAnalyzer:
    def __init__(self, video_path, cfg, verbose=False, audit_mode=False):
        self.cfg = cfg
        self.video_path = video_path
        self.verbose = verbose
        self.audit_mode = audit_mode
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        inf = cfg.get("inference", {})
        self.device = inf.get("device")
        self.imgsz = inf.get("imgsz", 640)
        self.half = inf.get("half", False)
        self.skip_frames = max(1, inf.get("skip_frames", 1))
        self.frame_counter = 0
        self.last_objects = {}

        self.model = YOLO(cfg["detection"]["model"])
        self.conf_thresh = cfg["detection"].get("confidence_threshold", 0.5)
        self.iou_thresh = cfg["detection"].get("iou_threshold", 0.45)

        cls_cfg = cfg.get("classes", {})
        self.class_mode = cls_cfg.get("mode", "coco")
        self.custom_classes = cls_cfg.get("custom", [])
        if self.class_mode == "custom" and self.custom_classes:
            try:
                self.model.set_classes(self.custom_classes)
                print(f"[INFO] Zero-shot классы: {self.custom_classes}")
            except Exception as e:
                print(f"[WARN] Не удалось установить custom classes: {e}")

        tracker_cfg = cfg.get("tracking", {}).get("tracker")
        self.use_builtin = bool(tracker_cfg)
        self.tracker_yaml = tracker_cfg
        if not self.use_builtin:
            self.tracker = SimpleTracker(
                distance_thresh=cfg["tracking"]["distance_threshold"],
                max_disappear=cfg["tracking"]["max_disappear"],
            )
        else:
            self.tracker = None

        self.interactions_enabled = cfg.get("interactions", {}).get("enabled", False)
        self.pose_model = None
        self.interaction_tracker = None
        if self.interactions_enabled:
            pose_path = cfg.get("interactions", {}).get("pose_model", "yolov8n-pose.pt")
            print(f"[INFO] HOI включён. Pose model: {pose_path}")
            self.pose_model = YOLO(pose_path)
            self.interaction_tracker = InteractionTracker(cfg)

        self.session = None
        self.frame_count = 0
        self.writer = None
        self.annotated_path = None
        self._seen_ids = set()
        self.audit_stats = defaultdict(lambda: {
            "count": 0, "frames": 0, "confidences": [],
            "first_frame": None, "last_frame": None,
            "example_ids": set(),
        })

    def _is_infer_frame(self):
        self.frame_counter += 1
        return (self.frame_counter % self.skip_frames == 0)

    def _process_builtin(self, frame):
        do_infer = self._is_infer_frame()
        if not do_infer:
            return self.last_objects
        results = self.model.track(
            frame, verbose=False, persist=True,
            tracker=self.tracker_yaml,
            conf=self.conf_thresh, iou=self.iou_thresh,
            device=self.device, imgsz=self.imgsz,
        )
        objects = {}
        if not results or not results[0].boxes:
            self.last_objects = objects
            return objects
        for box in results[0].boxes:
            tid = box.id
            if tid is None:
                continue
            tid = int(tid)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            objects[tid] = {
                "id": tid, "class": cls_name,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "conf": conf, "active": True,
            }
            if self.audit_mode:
                st = self.audit_stats[cls_name]
                st["count"] += 1
                st["confidences"].append(conf)
                st["example_ids"].add(tid)
                if st["first_frame"] is None:
                    st["first_frame"] = self.frame_count
                st["last_frame"] = self.frame_count
        self.last_objects = objects
        return objects

    def _process_simple(self, frame):
        do_infer = self._is_infer_frame()
        if not do_infer:
            return self.last_objects
        results = self.model(frame, verbose=False, conf=self.conf_thresh, iou=self.iou_thresh,
                             device=self.device, imgsz=self.imgsz)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = self.model.names[int(box.cls)]
                detections.append((x1, y1, x2, y2, cls_name, conf))
                if self.audit_mode:
                    st = self.audit_stats[cls_name]
                    st["count"] += 1
                    st["confidences"].append(conf)
                    st["first_frame"] = self.frame_count if st["first_frame"] is None else st["first_frame"]
                    st["last_frame"] = self.frame_count
        objects = self.tracker.update(detections)
        if self.audit_mode:
            for oid, obj in objects.items():
                self.audit_stats[obj["class"]]["example_ids"].add(oid)
        self.last_objects = objects
        return objects

    def _process_frame(self, frame):
        return self._process_builtin(frame) if self.use_builtin else self._process_simple(frame)

    def _extract_poses(self, results):
        poses = {}
        if results.keypoints is None:
            return poses
        for kpts, box in zip(results.keypoints.xy, results.boxes):
            tid = int(box.id) if box.id is not None else None
            if tid is None:
                continue
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            poses[tid] = {
                "bbox": (x1, y1, x2, y2),
                "keypoints": kpts.cpu().numpy(),
            }
        return poses

    def _draw(self, frame, objects, interactions=None, interpolated=False):
        overlay = frame.copy()
        recording = self.session and self.session.is_recording
        for oid, obj in objects.items():
            if not obj.get("active", False):
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["class"]
            is_target = recording and cls.lower() in self.session.target_classes
            color = (0, 255, 0) if is_target else (0, 165, 255)
            label = f"{cls} #{oid}"
            if interpolated:
                label += " [i]"
            if recording and is_target:
                trec = self.session.targets.get(oid)
                if trec:
                    live_dur = trec["total_duration"] + (time.time() - trec["last_seen"])
                    label += f" | {live_dur:.1f}s"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if interactions:
            overlay = self.interaction_tracker.draw(overlay, interactions)
        return overlay

    def _log_new_objects(self, objects):
        if not self.verbose:
            return
        for oid, obj in objects.items():
            if not obj.get("active", False):
                continue
            if oid not in self._seen_ids:
                self._seen_ids.add(oid)
                cls = obj["class"]
                is_target = cls.lower() in self.session.target_classes
                t = "ЦЕЛЕВОЙ" if is_target else "посторонний"
                print(f"  [NEW] {t:>12} | {cls:<15} | ID #{oid}")

    def run(self, target_classes, expected_seconds=None, save_video=False):
        self.session = Session(target_classes, expected_seconds)
        if save_video:
            out_dir = Path(self.cfg["export"]["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            self.annotated_path = out_dir / f"annotated_{Path(self.video_path).stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.annotated_path), fourcc, self.fps, (self.width, self.height))
            print(f"[INFO] Аннотированное видео: {self.annotated_path}")

        print(f"[INFO] Файл: {self.video_path}")
        print(f"[INFO] Кадров: {self.total_frames}, FPS: {self.fps:.1f}, "
              f"Длительность: {self.total_frames/self.fps:.1f}с")
        print(f"[INFO] Целевые: {target_classes}")
        if expected_seconds:
            print(f"[INFO] Норматив: {expected_seconds} сек")
        print(f"[INFO] Параметры инференса: device={self.device}, imgsz={self.imgsz}, "
              f"half={self.half}, skip_frames={self.skip_frames}")
        if self.verbose:
            print("[INFO] Verbose mode ON")
        if self.audit_mode:
            print("[INFO] AUDIT mode ON")
        if self.interactions_enabled:
            print("[INFO] HOI mode ON")
        print("[INFO] Обработка начата...")

        start_time = time.time()
        last_print = start_time
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame_count += 1
            objects = self._process_frame(frame)
            self.session.update(objects)
            self._log_new_objects(objects)

            interactions = None
            if self.interactions_enabled and self.pose_model and self._is_infer_frame():
                pose_results = self.pose_model.track(
                    frame, verbose=False, persist=True,
                    tracker="bytetrack.yaml",
                    device=self.device, imgsz=self.imgsz,
                )[0]
                poses = self._extract_poses(pose_results)
                interactions = self.interaction_tracker.update(poses, objects, frame.shape)

            if self.writer:
                self.writer.write(self._draw(frame, objects, interactions=interactions,
                                              interpolated=(self.frame_count % self.skip_frames != 0)))

            now = time.time()
            if now - last_print >= 5.0:
                pct = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
                proc_fps = self.frame_count / (now - start_time)
                print(f"  {self.frame_count}/{self.total_frames} ({pct:.1f}%) @ {proc_fps:.1f} FPS")
                last_print = now

        self.cap.release()
        if self.writer:
            self.writer.release()
        elapsed = time.time() - start_time
        print(f"[INFO] Готово за {elapsed:.1f} сек, средний FPS: {self.frame_count/elapsed:.1f}")
        return self.session.stop()

    def save_reports(self, summary, exporter):
        for fmt in self.cfg["export"]["formats"]:
            if fmt == "csv":
                data = exporter.to_csv(summary); ext = "csv"
            elif fmt == "json":
                data = exporter.to_json(summary); ext = "json"
            elif fmt == "xlsx":
                data = exporter.to_xlsx(summary); ext = "xlsx"
            else:
                continue
            filename = exporter._make_filename(summary, ext)
            with open(filename, "wb") as f:
                f.write(data)
            print(f"[OK] {filename}")

    def print_audit_report(self):
        if not self.audit_mode:
            return
        print("" + "=" * 70)
        print("AUDIT: ВСЕ ОБЪЕКТЫ, КОТОРЫЕ МОДЕЛЬ УВИДЕЛА НА ВИДЕО")
        print("=" * 70)
        if not self.audit_stats:
            print("  (ничего не обнаружено)")
            return
        sorted_items = sorted(self.audit_stats.items(), key=lambda x: x[1]["count"], reverse=True)
        print(f"{'Класс':<18} {'Детекций':>10} {'Уник.ID':>8} {'Средн.conf':>10} {'Первый кадр':>12}")
        print("-" * 70)
        for cls, st in sorted_items:
            avg_conf = sum(st["confidences"]) / len(st["confidences"]) if st["confidences"] else 0
            print(f"{cls:<18} {st['count']:>10} {len(st['example_ids']):>8} "
                  f"{avg_conf:>9.2f} {st['first_frame'] or 0:>12}")
        print("=" * 70)
        print(f"Всего уникальных классов: {len(sorted_items)}")


def print_model_info(cfg):
    model_path = cfg["detection"]["model"]
    cls_cfg = cfg.get("classes", {})
    mode = cls_cfg.get("mode", "coco")
    print("=" * 70)
    print("ИНФОРМАЦИЯ О ТЕКУЩЕЙ МОДЕЛИ")
    print("=" * 70)
    print(f"Модель: {model_path}")
    print(f"Режим классов: {mode}")
    if mode == "coco":
        print("Стандартные COCO-классы (80 шт.) — полный список см. README.md")
        print("Если нужного класса нет — переключитесь на YOLO-World.")
    elif mode == "custom":
        custom = cls_cfg.get("custom", [])
        print(f"Zero-shot классы ({len(custom)} шт.):")
        for c in custom:
            print(f"  {c}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Анализ видеофайла на объекты")
    parser.add_argument("--video", "-v", type=str, default=None, help="Путь к видеофайлу")
    parser.add_argument("--targets", "-t", type=str, default=None, help="Целевые классы через запятую")
    parser.add_argument("--expected", "-e", type=float, default=None, help="Норматив, сек")
    parser.add_argument("--save-video", "-s", action="store_true", help="Сохранить аннотированное видео")
    parser.add_argument("--verbose", "-V", action="store_true", help="Логировать новые объекты")
    parser.add_argument("--audit", "-a", action="store_true", help="Режим аудита")
    parser.add_argument("--info", "-i", action="store_true", help="Информация о модели")
    args = parser.parse_args()

    cfg = load_config()
    if args.info:
        print_model_info(cfg)
        return

    video_path = args.video or cfg["detection"]["source"]
    if video_path in (0, "0", "webcam"):
        print("Ошибка: укажите путь к видеофайлу")
        return

    if args.targets:
        target_classes = [c.strip().lower() for c in args.targets.split(",") if c.strip()]
    else:
        target_classes = cfg["targets"]["classes"]
    expected = args.expected if args.expected is not None else cfg["targets"].get("expected_seconds")

    analyzer = BatchVideoAnalyzer(video_path, cfg, verbose=args.verbose, audit_mode=args.audit)
    exporter = Exporter(cfg)

    summary = analyzer.run(target_classes, expected, save_video=args.save_video)
    analyzer.save_reports(summary, exporter)

    if analyzer.interactions_enabled and analyzer.interaction_tracker:
        summary["interactions"] = analyzer.interaction_tracker.get_summary()
        if summary["interactions"]:
            print("" + "=" * 70)
            print("ВЗАИМОДЕЙСТВИЯ (Human-Object Interaction)")
            print("=" * 70)
            print(f"{'ID':<5} {'Человек':<8} {'Предмет':<8} {'Класс':<12} {'Операция':<20} {'С':<10} {'По':<10} {'Длительность'}")
            print("-" * 70)
            for r in summary["interactions"]:
                print(f"{r['interaction_id']:<5} {r['person_id']:<8} {r['object_id']:<8} "
                      f"{r['object_class']:<12} {r['label']:<20} {r['start_time']:<10} "
                      f"{r['end_time']:<10} {r['duration_sec']}с")
            print("=" * 70)

    analyzer.print_audit_report()

    print("" + "=" * 70)
    print("СВОДКА ПО СЕССИИ")
    print("=" * 70)
    print(f"Период:       {summary['session_start']} — {summary['session_end']}")
    print(f"Длительность: {summary['session_duration_sec']} сек")
    print(f"Целевые:      {', '.join(summary['target_classes'])}")
    if summary['expected_seconds']:
        print(f"Норматив:     {summary['expected_seconds']} сек")
    print(f"--- ЦЕЛЕВЫЕ ОБЪЕКТЫ ({len(summary['targets'])}) ---")
    for r in summary['targets']:
        extra = ""
        if 'status' in r:
            extra = f" | статус: {r['status']} (отклонение {r.get('deviation_sec', '-')}с)"
        print(f"  • {r['class']:<15} #{r['tracker_id']:<4} | {r['total_duration_sec']:>6}с "
              f"(с {r['first_seen']} по {r['last_seen']}){extra}")
    if summary['ignored_summary']:
        print(f"--- ПОСТОРОННИЕ ОБЪЕКТЫ ---")
        for r in summary['ignored_summary']:
            print(f"  • {r['class']:<15} | {r['count']} шт. | суммарно {r['total_duration_sec']}с")
    all_objects = summary['targets'] + summary['ignored']
    all_objects.sort(key=lambda x: x['first_seen'])
    print(f"--- ВСЕ ОБЪЕКТЫ В КАДРЕ (всего {len(all_objects)}) ---")
    print(f"{'ID':<5} {'Класс':<15} {'Тип':<12} {'С':<10} {'По':<10} {'Длительность'}")
    print("-" * 60)
    for r in all_objects:
        obj_type = "ЦЕЛЕВОЙ" if r['object_type'] == 'target' else "посторонний"
        print(f"{r['tracker_id']:<5} {r['class']:<15} {obj_type:<12} "
              f"{r['first_seen']:<10} {r['last_seen']:<10} {r['total_duration_sec']}с")
    print("=" * 70)


if __name__ == "__main__":
    main()