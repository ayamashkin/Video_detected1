#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Видеодетектор с сессионным контролем + zero-shot + Human-Object Interaction (HOI).

Поддерживает:
  • webcam, IP-камеры, видеофайлы
  • COCO / YOLO-World объекты
  • YOLOv8-pose скелеты
  • Взаимодействия: человек + предмет = операция (по правилам config.yaml)

Запуск:
    python app.py
    -> http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from core import load_config, Session, Exporter, SimpleTracker
from interaction_tracker import InteractionTracker


CONFIG = load_config()


class VideoProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        src = cfg["detection"]["source"]
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg["detection"].get("buffer_size", 1))

        src_str = str(src)
        self.is_file = (isinstance(src, str)
                        and not src_str.isdigit()
                        and not src_str.lower().startswith(("rtsp://", "http://", "https://")))
        self.source_fps = None
        if self.is_file:
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps and self.source_fps > 0:
                print(f"[INFO] Видеофайл: {src}, FPS: {self.source_fps:.1f}")
            else:
                self.source_fps = 25.0
                print(f"[INFO] Видеофайл: {src}, FPS не определён, используем 25")

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
        self.use_builtin_tracker = bool(tracker_cfg)
        self.tracker_yaml = tracker_cfg
        if not self.use_builtin_tracker:
            self.tracker = SimpleTracker(
                distance_thresh=cfg["tracking"]["distance_threshold"],
                max_disappear=cfg["tracking"]["max_disappear"],
            )
        else:
            self.tracker = None

        # ── Human-Object Interaction ──
        self.interactions_enabled = cfg.get("interactions", {}).get("enabled", False)
        self.pose_model = None
        self.interaction_tracker = None
        if self.interactions_enabled:
            pose_path = cfg.get("interactions", {}).get("pose_model", "yolov8n-pose.pt")
            print(f"[INFO] HOI включён. Pose model: {pose_path}")
            self.pose_model = YOLO(pose_path)
            self.interaction_tracker = InteractionTracker(cfg)

        self.session = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

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
        objects = self.tracker.update(detections)
        self.last_objects = objects
        return objects

    def _extract_poses(self, results):
        """Извлекает позы из YOLOv8-pose track."""
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

    def _loop(self):
        while self.running:
            frame_start = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    print("[INFO] Видеофайл завершён")
                time.sleep(0.5)
                continue

            if self.use_builtin_tracker:
                tracker_objects = self._process_builtin(frame)
            else:
                tracker_objects = self._process_simple(frame)

            if self.session and self.session.is_recording:
                self.session.update(tracker_objects)

            # ── Human-Object Interaction ──
            interactions = []
            if self.interactions_enabled and self.pose_model and self._is_infer_frame():
                pose_results = self.pose_model.track(
                    frame, verbose=False, persist=True,
                    tracker="bytetrack.yaml",
                    device=self.device, imgsz=self.imgsz,
                )[0]
                poses = self._extract_poses(pose_results)
                interactions = self.interaction_tracker.update(poses, tracker_objects, frame.shape)

            overlay = frame.copy()
            recording = self.session and self.session.is_recording
            for oid, obj in tracker_objects.items():
                if not obj.get("active", False):
                    continue
                x1, y1, x2, y2 = obj["bbox"]
                cls = obj["class"]
                is_target = recording and cls.lower() in self.session.target_classes
                color = (0, 255, 0) if is_target else (0, 165, 255)
                label = f"{cls} #{oid}"
                if not self._is_infer_frame():
                    label += " [i]"
                if recording and is_target:
                    trec = self.session.targets.get(oid)
                    if trec:
                        live_dur = trec["total_duration"] + (time.time() - trec["last_seen"])
                        label += f" | {live_dur:.1f}s"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Рисуем взаимодействия
            if interactions:
                overlay = self.interaction_tracker.draw(overlay, interactions)

            if recording:
                elapsed = time.time() - self.session.start_time
                timer_text = f"REC {int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
                cv2.circle(overlay, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(overlay, timer_text, (50, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if self.skip_frames > 1:
                    cv2.putText(overlay, f"skip={self.skip_frames}", (50, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            with self.lock:
                self.frame = overlay.copy()

            if self.is_file and self.source_fps:
                process_time = time.time() - frame_start
                sleep_time = max(0.0, 1.0 / self.source_fps - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def get_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self.frame)
            return buf.tobytes() if ok else None

    def start_session(self, target_classes=None, expected_seconds=None):
        if self.session and self.session.is_recording:
            raise RuntimeError("Сессия уже запущена")
        if target_classes is None:
            target_classes = self.cfg["targets"]["classes"]
        if expected_seconds is None:
            expected_seconds = self.cfg["targets"].get("expected_seconds")
        self.session = Session(target_classes, expected_seconds)
        return {"status": "started", "start_time": self.session.start_time}

    def stop_session(self):
        if not self.session or not self.session.is_recording:
            raise RuntimeError("Нет активной сессии")
        summary = self.session.stop()
        if self.interactions_enabled and self.interaction_tracker:
            summary["interactions"] = self.interaction_tracker.get_summary()
        return summary

    def session_status(self):
        if not self.session:
            return {"recording": False, "elapsed": 0}
        if self.session.is_recording:
            elapsed = round(time.time() - self.session.start_time, 1)
            return {
                "recording": True, "elapsed": elapsed,
                "target_classes": sorted(list(self.session.target_classes)),
            }
        return {"recording": False, "elapsed": round(self.session.end_time - self.session.start_time, 1)}

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)
        self.cap.release()


app = Flask(__name__)
processor = None
exporter = None

@app.route("/")
def index():
    return render_template("index.html", config=CONFIG)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            jpeg = processor.get_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    target_classes = data.get("target_classes")
    expected = data.get("expected_seconds")
    if expected is not None:
        try:
            expected = float(expected)
        except ValueError:
            return jsonify({"error": "expected_seconds must be number"}), 400
    try:
        result = processor.start_session(target_classes, expected)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409

@app.route("/api/stop", methods=["POST"])
def api_stop():
    try:
        result = processor.stop_session()
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409

@app.route("/api/status")
def api_status():
    return jsonify(processor.session_status())

@app.route("/api/export/<fmt>", methods=["GET"])
def api_export(fmt):
    if not processor.session:
        return jsonify({"error": "Нет данных для экспорта"}), 404
    summary = processor.session._summary()
    if processor.interactions_enabled and processor.interaction_tracker:
        summary["interactions"] = processor.interaction_tracker.get_summary()
    if fmt == "csv":
        data = exporter.to_csv(summary); mime = "text/csv; charset=utf-8-sig"; ext = "csv"
    elif fmt == "json":
        data = exporter.to_json(summary); mime = "application/json; charset=utf-8"; ext = "json"
    elif fmt == "xlsx":
        data = exporter.to_xlsx(summary); mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"; ext = "xlsx"
    else:
        return jsonify({"error": f"Неизвестный формат: {fmt}"}), 400
    filename = exporter._make_filename(summary, ext)
    with open(filename, "wb") as f:
        f.write(data)
    return Response(data, mimetype=mime,
                    headers={"Content-Disposition": f"attachment; filename={filename.name}"})

@app.route("/api/config")
def api_config():
    inf = CONFIG.get("inference", {})
    return jsonify({
        "detection": {
            "model": CONFIG["detection"]["model"],
            "confidence_threshold": CONFIG["detection"]["confidence_threshold"],
        },
        "classes": {
            "mode": CONFIG.get("classes", {}).get("mode", "coco"),
            "custom": CONFIG.get("classes", {}).get("custom", []),
        },
        "targets": CONFIG["targets"],
        "tracking": {"tracker": CONFIG.get("tracking", {}).get("tracker")},
        "inference": {
            "device": inf.get("device"),
            "imgsz": inf.get("imgsz", 640),
            "half": inf.get("half", False),
            "skip_frames": inf.get("skip_frames", 1),
        },
        "interactions": {
            "enabled": CONFIG.get("interactions", {}).get("enabled", False),
            "pose_model": CONFIG.get("interactions", {}).get("pose_model", "yolov8n-pose.pt"),
        },
        "export": {
            "formats": CONFIG["export"]["formats"],
            "output_dir": str(Path(CONFIG["export"]["output_dir"]).resolve()),
        },
        "ui": CONFIG.get("ui", {}),
    })

if __name__ == "__main__":
    processor = VideoProcessor(CONFIG)
    exporter = Exporter(CONFIG)
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        processor.stop()