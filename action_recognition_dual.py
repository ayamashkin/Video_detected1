#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Двухмодельный анализатор: COCO (yolov8n.pt) + Custom (best.pt CWD1500) + Pose (HOI).

COCO-модель: person, car, laptop, tv, cup, bottle и т.д. (80 классов)
Custom-модель: wheel (из CWD1500) — точнее, чем COCO

Использование:
    python action_recognition_dual.py --video data/video.mp4 --custom-model models/best.pt
    python action_recognition_dual.py --video data/video.mp4 --custom-model models/best.pt --no-display
"""

import argparse
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from core import load_config, translate_object, translate_action, put_text_ru
from interaction_tracker import InteractionTracker


# ═══════════════════════════════════════════════════════
#  Классификация действий (копия из оригинала)
# ═══════════════════════════════════════════════════════
def classify_action(keypoints, nearby_objects=None, rules=None):
    if keypoints is None or len(keypoints) < 17:
        return "unknown", 0.0
    r = rules or {}
    phone_rule = r.get("using_phone", {})
    walk_rule = r.get("walking", {})
    sit_rule = r.get("sitting", {})
    work_rule = r.get("working", {})
    stand_conf = r.get("standing", {}).get("confidence", 0.60)

    hips = (keypoints[11] + keypoints[12]) / 2.0
    k = keypoints - hips
    shoulder_dist = np.linalg.norm(keypoints[5] - keypoints[6])
    if shoulder_dist > 0:
        k = k / (shoulder_dist + 1e-6)

    y_spread = np.ptp(k[:, 1])
    x_spread = np.ptp(k[:, 0])
    wrist_y = k[[9, 10], 1]
    wrist_up = np.mean(wrist_y < -0.3)
    hip_var = np.var(k[[11, 12], 0])

    wup_thr = phone_rule.get("wrist_up_threshold", 0.6)
    if wrist_up > wup_thr:
        phone_nearby = False
        phone_classes = [c.lower() for c in phone_rule.get("phone_classes", ["cell phone", "phone", "smartphone"])]
        phone_dist = phone_rule.get("phone_distance_px", 200)
        if nearby_objects:
            for obj_cls, dist in nearby_objects:
                if obj_cls.lower() in phone_classes and dist < phone_dist:
                    phone_nearby = True
                    break
        if phone_nearby:
            return "using_phone", phone_rule.get("confidence", 0.85)
        if wrist_up > 0.8:
            return "working", work_rule.get("confidence", 0.70)
        return "standing", stand_conf

    if hip_var > walk_rule.get("hip_var_threshold", 0.05) and x_spread > walk_rule.get("x_spread_threshold", 0.8):
        return "walking", walk_rule.get("confidence", 0.80)
    if y_spread < sit_rule.get("y_spread_threshold", 0.35) and hip_var < sit_rule.get("hip_var_threshold", 0.02):
        return "sitting", sit_rule.get("confidence", 0.85)
    if wrist_up > work_rule.get("wrist_up_threshold", 0.25) and hip_var < work_rule.get("hip_var_threshold", 0.03):
        return "working", work_rule.get("confidence", 0.70)
    return "standing", stand_conf


SKELETON_LINES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, keypoints, color=(0, 255, 255)):
    for i, j in SKELETON_LINES:
        pt1 = tuple(keypoints[i].astype(int))
        pt2 = tuple(keypoints[j].astype(int))
        if pt1[0] == 0 and pt1[1] == 0:
            continue
        if pt2[0] == 0 and pt2[1] == 0:
            continue
        cv2.line(frame, pt1, pt2, color, 2)
    for i, (x, y) in enumerate(keypoints):
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame


# ═══════════════════════════════════════════════════════
#  Объединение результатов двух моделей
# ═══════════════════════════════════════════════════════
def merge_detections(coco_results, custom_results, custom_classes, id_offset=1000):
    """
    Объединяет детекции COCO и Custom.
    Custom-классы (например, wheel) берутся из custom_results.
    Остальное — из coco_results.
    ID custom объектов сдвигаются на id_offset, чтобы не конфликтовать.
    """
    objects = {}

    # COCO-объекты
    if coco_results.boxes is not None:
        for box in coco_results.boxes:
            tid = box.id
            if tid is None:
                continue
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = coco_results.names[int(box.cls)]
            conf = float(box.conf)
            objects[tid] = {
                "bbox": (x1, y1, x2, y2),
                "class": cls,
                "conf": conf,
                "active": True,
                "source": "coco",
            }

    # Custom-объекты (только указанные классы)
    custom_classes_lower = [c.lower() for c in custom_classes]
    if custom_results.boxes is not None:
        for box in custom_results.boxes:
            tid = box.id
            if tid is None:
                continue
            tid = int(tid) + id_offset  # сдвиг ID, чтобы не конфликтовать с COCO
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = custom_results.names[int(box.cls)]
            conf = float(box.conf)

            # Берём только нужные классы из custom
            if cls.lower() not in custom_classes_lower:
                continue

            objects[tid] = {
                "bbox": (x1, y1, x2, y2),
                "class": cls,
                "conf": conf,
                "active": True,
                "source": "custom",
            }

    return objects


def main():
    parser = argparse.ArgumentParser(description="Двухмодельный: COCO + Custom + Pose + HOI")
    parser.add_argument("--video", "-v", required=True, help="Путь к видеофайлу")
    parser.add_argument("--config", "-c", default="config.yaml", help="config.yaml")
    parser.add_argument("--coco-model", default="models/yolov8n.pt", help="COCO модель (80 классов)")
    parser.add_argument("--custom-model", default="models/best.pt", help="Custom модель (CWD1500)")
    parser.add_argument("--custom-classes", default="wheel", help="Классы из custom-модели (через запятую)")
    parser.add_argument("--window", "-w", type=int, default=30, help="Не используется (совместимость)")
    parser.add_argument("--save-video", "-s", action="store_true", help="Сохранить аннотированное видео")
    parser.add_argument("--save-report", "-r", action="store_true", help="Сохранить отчёт")
    parser.add_argument("--verbose", "-V", action="store_true", help="Вывод каждого кадра")
    parser.add_argument("--output-dir", "-o", default="./output", help="Папка для отчётов")
    parser.add_argument("--no-display", action="store_true", help="Без окна OpenCV (ускоряет на CPU)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_classes = [c.strip() for c in args.custom_classes.split(",")]

    inf = cfg.get("inference", {})
    device = inf.get("device")
    imgsz = inf.get("imgsz", 640)
    skip_frames = max(1, inf.get("skip_frames", 1))
    conf_thresh = cfg["detection"].get("confidence_threshold", 0.5)
    iou_thresh = cfg["detection"].get("iou_threshold", 0.45)

    # ── Загрузка моделей ──
    print(f"[INFO] COCO модель: {args.coco_model}")
    coco_model = YOLO(args.coco_model)

    print(f"[INFO] Custom модель: {args.custom_model}")
    print(f"[INFO] Custom классы: {custom_classes}")
    custom_model = YOLO(args.custom_model)

    # Pose / HOI
    pose_model = None
    interaction_tracker = None
    hoi_enabled = cfg.get("interactions", {}).get("enabled", False)
    if hoi_enabled:
        pose_path = cfg.get("interactions", {}).get("pose_model", "yolov8n-pose.pt")
        print(f"[INFO] HOI включён. Pose model: {pose_path}")
        pose_model = YOLO(pose_path)
        interaction_tracker = InteractionTracker(cfg)

    # ── Видео ──
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Файл: {args.video}")
    print(f"[INFO] {total_frames} кадров, {fps:.1f} FPS, {w}x{h}")
    print(f"[INFO] skip_frames: {skip_frames}")
    print(f"[INFO] imgsz: {imgsz}")
    print(f"[INFO] Начало обработки...")

    if not args.no_display:
        cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Detection", 1280, 720)
        cv2.startWindowThread()
        print("[INFO] Окно создано. Нажмите Q или ESC для выхода.")
    else:
        print("[INFO] Режим без окна (--no-display).")

    writer = None
    if args.save_video:
        vid_path = out_dir / f"annotated_{Path(args.video).stem}.mp4"
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[INFO] Сохранение видео: {vid_path}")

    frame_count = 0
    infer_counter = 0
    start_time = time.time()
    last_print = start_time

    last_objects = {}
    last_poses = {}
    last_actions = {}

    object_history = defaultdict(lambda: {"first": None, "last": None, "frames": 0, "class": ""})
    person_history = defaultdict(lambda: {"first": None, "last": None, "frames": 0, "actions": defaultdict(int)})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        now = time.time()

        do_infer = (frame_count % skip_frames == 0)

        if do_infer:
            infer_counter += 1

            # ── 1. COCO детекция ──
            coco_res = coco_model.track(
                frame, verbose=False, persist=True,
                tracker=cfg.get("tracking", {}).get("tracker", "bytetrack.yaml"),
                conf=conf_thresh, iou=iou_thresh,
                device=device, imgsz=imgsz,
            )[0]

            # ── 2. Custom детекция ──
            custom_res = custom_model.track(
                frame, verbose=False, persist=True,
                tracker=cfg.get("tracking", {}).get("tracker", "bytetrack.yaml"),
                conf=conf_thresh, iou=iou_thresh,
                device=device, imgsz=imgsz,
            )[0]

            # ── 3. Объединение ──
            objects = merge_detections(coco_res, custom_res, custom_classes, id_offset=1000)

            # Статистика
            for tid, obj in objects.items():
                oh = object_history[tid]
                if oh["first"] is None:
                    oh["first"] = frame_count
                oh["last"] = frame_count
                oh["frames"] += 1
                oh["class"] = obj["class"]

            last_objects = objects

            # ── 4. Pose (HOI) ──
            poses = {}
            actions = {}
            if pose_model is not None:
                pose_res = pose_model.track(
                    frame, verbose=False, persist=True,
                    tracker="bytetrack.yaml",
                    device=device, imgsz=imgsz,
                )[0]

                if pose_res.keypoints is not None and pose_res.boxes is not None:
                    for kpts, box in zip(pose_res.keypoints.xy, pose_res.boxes):
                        tid = int(box.id) if box.id is not None else None
                        if tid is None:
                            continue
                        tid = int(tid)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        kp = kpts.cpu().numpy()
                        poses[tid] = {"bbox": (x1, y1, x2, y2), "keypoints": kp}

                    rules = cfg.get("action_rules", {})
                    for tid, pdata in poses.items():
                        kp = pdata["keypoints"]
                        nearby = []
                        for oid, odata in objects.items():
                            if not odata.get("active", False):
                                continue
                            wrists = [kp[9], kp[10]]
                            cx = (odata["bbox"][0] + odata["bbox"][2]) / 2.0
                            cy = (odata["bbox"][1] + odata["bbox"][3]) / 2.0
                            for w in wrists:
                                if w[0] == 0 and w[1] == 0:
                                    continue
                                d = np.hypot(w[0] - cx, w[1] - cy)
                                if d < 250:
                                    nearby.append((odata["class"], d))
                        action, conf = classify_action(kp, nearby, rules)
                        actions[tid] = (action, conf)

                    for tid, pdata in poses.items():
                        x1, y1, x2, y2 = pdata["bbox"]
                        kp = pdata["keypoints"]
                        action, conf = actions.get(tid, ("unknown", 0.0))
                        ph = person_history[tid]
                        if ph["first"] is None:
                            ph["first"] = frame_count
                        ph["last"] = frame_count
                        ph["frames"] += 1
                        ph["actions"][action] += 1
            last_poses = poses
            last_actions = actions
        else:
            objects = last_objects
            poses = last_poses
            actions = last_actions

        # ── Отрисовка ──
        overlay = frame.copy()
        for tid, obj in objects.items():
            if not obj.get("active", False):
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["class"]
            is_target = cls.lower() in [c.lower() for c in cfg.get("targets", {}).get("classes", [])]
            color = (0, 255, 0) if is_target else (0, 165, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{translate_object(cls)} #{tid}"
            if not do_infer:
                label += " [i]"
            if obj.get("source") == "custom":
                label += " [C]"
            overlay = put_text_ru(overlay, label, (x1, max(y1 - 10, 20)), font_size=16, color=color, thickness=2)

        for tid, pdata in poses.items():
            x1, y1, x2, y2 = pdata["bbox"]
            kp = pdata["keypoints"]
            action, conf = actions.get(tid, ("unknown", 0.0))
            overlay = draw_skeleton(overlay, kp)
            label = f"#{tid} {translate_action(action)} ({conf:.2f})"
            overlay = put_text_ru(overlay, label, (x1, max(y1 - 10, 20)), font_size=18, color=(255, 0, 255), thickness=2)

        interactions = []
        if interaction_tracker is not None and poses and objects and do_infer:
            interactions = interaction_tracker.update(poses, objects, overlay.shape)
            for inter in interactions:
                px1, py1, px2, py2 = inter["bbox_person"]
                ox1, oy1, ox2, oy2 = inter["bbox_object"]
                pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
                ocx, ocy = int((ox1 + ox2) / 2), int((oy1 + oy2) / 2)
                cv2.line(overlay, (pcx, pcy), (ocx, ocy), (255, 0, 255), 2)
                label = f"{inter['label']} ({inter['duration_sec']}s)"
                mx, my = (pcx + ocx) // 2, (pcy + ocy) // 2
                overlay = put_text_ru(overlay, label, (mx, max(my - 10, 20)), font_size=16, color=(255, 0, 255), thickness=2)

        if not args.no_display:
            display_scale = min(1.0, 960 / max(overlay.shape[1], overlay.shape[0]))
            if display_scale < 1.0:
                disp_w = int(overlay.shape[1] * display_scale)
                disp_h = int(overlay.shape[0] * display_scale)
                display_frame = cv2.resize(overlay, (disp_w, disp_h))
            else:
                display_frame = overlay
            cv2.imshow("Video Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Прервано пользователем")
                break

        if writer:
            writer.write(overlay)

        # ── Консольный вывод ──
        do_print = (now - last_print >= 5.0) or args.verbose
        if do_print and (objects or poses):
            elapsed = now - start_time
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            proc_fps = frame_count / elapsed if elapsed > 0 else 0
            infer_fps = infer_counter / elapsed if elapsed > 0 else 0
            print(f"=== Frame {frame_count}/{total_frames} ({pct:.1f}%) | {elapsed:.1f}s | {proc_fps:.1f} FPS disp | {infer_fps:.1f} FPS infer ===")

            if objects:
                print("  [OBJ]  ", end="")
                parts = []
                for tid, obj in objects.items():
                    src = "C" if obj.get("source") == "custom" else ""
                    parts.append(f"{translate_object(obj['class'])}#{tid}({obj['conf']:.2f}){src}")
                print("  |  ".join(parts))

            if poses:
                for tid, pdata in poses.items():
                    act, conf = actions.get(tid, ("unknown", 0.0))
                    nearby = []
                    for oid, odata in objects.items():
                        kpts = pdata.get("keypoints")
                        if kpts is not None and len(kpts) >= 17:
                            wrists = [kpts[9], kpts[10]]
                            cx = (odata["bbox"][0] + odata["bbox"][2]) / 2
                            cy = (odata["bbox"][1] + odata["bbox"][3]) / 2
                            d = min(np.hypot(w[0]-cx, w[1]-cy) for w in wrists if not (w[0]==0 and w[1]==0))
                            if d < 200:
                                nearby.append(f"{translate_object(odata['class'])}#{oid}({d:.0f}px)")
                    nearby_str = ", ".join(nearby) if nearby else "—"
                    print(f"  [PERS] #{tid} {translate_action(act)}({conf:.2f}) | рядом: {nearby_str}")

            if interactions:
                for inter in interactions:
                    print(f"  [HOI]  человек#{inter['person_id']} + {translate_object(inter['object_class'])}#{inter['object_id']} -> '{inter['label']}' | {inter['duration_sec']}s")

            last_print = now

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"[INFO] Готово. {frame_count} кадров за {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")

    # ── Отчёт ──
    print("" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 80)

    if object_history:
        print(f"--- ОБЪЕКТЫ ({len(object_history)}) ---")
        print(f"{'ID':<6} {'Класс':<15} {'Источник':<8} {'Кадров':>8}")
        print("-" * 50)
        for tid, oh in sorted(object_history.items()):
            src = "custom" if tid >= 1000 else "coco"
            print(f"{tid:<6} {oh['class']:<15} {src:<8} {oh['frames']:>8}")

    if person_history:
        print(f"--- ЛЮДИ ({len(person_history)}) ---")
        print(f"{'ID':<6} {'Кадров':>8} {'Доминантное действие':<25}")
        print("-" * 70)
        for tid, ph in sorted(person_history.items()):
            dominant = max(ph["actions"].items(), key=lambda x: x[1])[0] if ph["actions"] else "unknown"
            print(f"{tid:<6} {ph['frames']:>8} {dominant:<25}")

    if interaction_tracker is not None:
        inter_summary = interaction_tracker.get_summary()
        if inter_summary:
            print(f"--- ВЗАИМОДЕЙСТВИЯ ({len(inter_summary)}) ---")
            for r in inter_summary:
                print(f"  {r['interaction_id']}: человек#{r['person_id']} + {r['object_class']}#{r['object_id']} -> '{r['label']}' {r['duration_sec']}s")
        else:
            print(f"--- ВЗАИМОДЕЙСТВИЯ: нет ---")
    print("=" * 80)

    if args.save_report:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report = {
            "video": args.video, "processed_at": ts,
            "frames": frame_count, "duration_sec": round(elapsed, 1),
            "objects": [{"id": tid, "class": oh["class"], "frames": oh["frames"], "source": "custom" if tid >= 1000 else "coco"}
                        for tid, oh in sorted(object_history.items())],
            "people": [{"id": tid, "frames": ph["frames"], "actions": dict(ph["actions"])}
                       for tid, ph in sorted(person_history.items())],
            "interactions": interaction_tracker.get_summary() if interaction_tracker else [],
        }
        json_path = out_dir / f"report_{Path(args.video).stem}_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] Отчёт: {json_path}")


if __name__ == "__main__":
    main()