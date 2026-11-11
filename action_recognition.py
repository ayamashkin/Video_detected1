#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комбинированный анализатор: объекты + действия людей + взаимодействия (HOI).
Оптимизирован для CPU: skip_frames синхронизирован, убран deprecated half.
"""

import argparse
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from core import load_config, translate_object, translate_action, put_text_ru
from interaction_tracker import InteractionTracker


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


def main():
    parser = argparse.ArgumentParser(description="Объекты + Действия + Взаимодействия")
    parser.add_argument("--video", "-v", required=True, help="Путь к видеофайлу")
    parser.add_argument("--config", "-c", default="config.yaml", help="config.yaml")
    parser.add_argument("--window", "-w", type=int, default=30, help="Не используется (совместимость)")
    parser.add_argument("--save-video", "-s", action="store_true", help="Сохранить аннотированное видео")
    parser.add_argument("--save-report", "-r", action="store_true", help="Сохранить отчёт в output/")
    parser.add_argument("--verbose", "-V", action="store_true", help="Вывод каждого кадра")
    parser.add_argument("--output-dir", "-o", default="./output", help="Папка для отчётов")
    parser.add_argument("--no-display", action="store_true", help="Без окна OpenCV (ускоряет на CPU)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_model_path = cfg["detection"]["model"]
    obj_model = YOLO(obj_model_path)
    conf_thresh = cfg["detection"].get("confidence_threshold", 0.5)
    iou_thresh = cfg["detection"].get("iou_threshold", 0.45)
    inf = cfg.get("inference", {})
    device = inf.get("device")
    imgsz = inf.get("imgsz", 640)
    skip_frames = max(1, inf.get("skip_frames", 1))

    cls_cfg = cfg.get("classes", {})
    if cls_cfg.get("mode") == "custom" and cls_cfg.get("custom"):
        try:
            obj_model.set_classes(cls_cfg["custom"])
            print(f"[INFO] Zero-shot классы: {cls_cfg['custom']}")
        except Exception as e:
            print(f"[WARN] set_classes: {e}")

    pose_model = None
    interaction_tracker = None
    hoi_enabled = cfg.get("interactions", {}).get("enabled", False)
    if hoi_enabled:
        pose_path = cfg.get("interactions", {}).get("pose_model", "yolov8n-pose.pt")
        print(f"[INFO] HOI включён. Pose model: {pose_path}")
        pose_model = YOLO(pose_path)
        interaction_tracker = InteractionTracker(cfg)

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
    print(f"[INFO] Объектная модель: {obj_model_path}")
    print(f"[INFO] skip_frames: {skip_frames} (анализ каждый {skip_frames}-й кадр)")
    print(f"[INFO] imgsz: {imgsz}")
    print(f"[INFO] Начало обработки...")

    if not args.no_display:
        cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Detection", 1280, 720)
        cv2.startWindowThread()
        print("[INFO] Окно 'Video Detection' создано. Нажмите Q или ESC для выхода.")
    else:
        print("[INFO] Режим без окна (--no-display). Обработка в фоне.")

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
            # ── 1. Детекция объектов (без half) ──
            obj_res = obj_model.track(
                frame, verbose=False, persist=True,
                tracker=cfg.get("tracking", {}).get("tracker", "bytetrack.yaml"),
                conf=conf_thresh, iou=iou_thresh,
                device=device, imgsz=imgsz,
            )[0]

            objects = {}
            if obj_res.boxes is not None:
                for box in obj_res.boxes:
                    tid = box.id
                    if tid is None:
                        continue
                    tid = int(tid)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = obj_res.names[int(box.cls)]
                    conf = float(box.conf)
                    objects[tid] = {
                        "bbox": (x1, y1, x2, y2), "class": cls,
                        "conf": conf, "active": True,
                    }
                    oh = object_history[tid]
                    if oh["first"] is None:
                        oh["first"] = frame_count
                    oh["last"] = frame_count
                    oh["frames"] += 1
                    oh["class"] = cls
            last_objects = objects

            # ── 2. Детекция поз (без half) ──
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

        overlay = frame.copy()
        for tid, obj in objects.items():
            if not obj.get("active", False):
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["class"]
            color = (0, 255, 0) if cls.lower() in [c.lower() for c in cfg.get("targets", {}).get("classes", [])] else (0, 165, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{translate_object(cls)} #{tid}"
            if not do_infer:
                label += " [i]"
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

        do_print = (now - last_print >= 5.0) or args.verbose
        if do_print and (objects or poses):
            elapsed = now - start_time
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            proc_fps = frame_count / elapsed if elapsed > 0 else 0
            infer_fps = infer_counter / elapsed if elapsed > 0 else 0
            print(f"=== Frame {frame_count}/{total_frames} ({pct:.1f}%) | {elapsed:.1f}s | {proc_fps:.1f} FPS (display) | {infer_fps:.1f} FPS (infer) ===")

            if objects:
                print("  [OBJ]  ", end="")
                parts = []
                for tid, obj in objects.items():
                    parts.append(f"{translate_object(obj['class'])}#{tid}({obj['conf']:.2f})")
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

    print("" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 80)

    if object_history:
        print(f"--- ОБЪЕКТЫ ({len(object_history)}) ---")
        print(f"{'ID':<6} {'Класс':<15} {'Кадров':>8} {'С':>8} {'По':>8}")
        print("-" * 50)
        for tid, oh in sorted(object_history.items()):
            print(f"{tid:<6} {oh['class']:<15} {oh['frames']:>8} {oh['first'] or 0:>8} {oh['last'] or 0:>8}")

    if person_history:
        print(f"--- ЛЮДИ ({len(person_history)}) ---")
        print(f"{'ID':<6} {'Кадров':>8} {'Доминантное действие':<25} {'Все действия'}")
        print("-" * 70)
        for tid, ph in sorted(person_history.items()):
            dominant = max(ph["actions"].items(), key=lambda x: x[1])[0] if ph["actions"] else "unknown"
            all_act = ", ".join(f"{a}:{c}" for a, c in sorted(ph["actions"].items(), key=lambda x: -x[1]))
            print(f"{tid:<6} {ph['frames']:>8} {dominant:<25} {all_act}")

    if interaction_tracker is not None:
        inter_summary = interaction_tracker.get_summary()
        if inter_summary:
            print(f"--- ВЗАИМОДЕЙСТВИЯ ({len(inter_summary)}) ---")
            print(f"{'ID':<5} {'Чел':<5} {'Предм':<6} {'Класс':<12} {'Операция':<22} {'С':<10} {'По':<10} {'Длит,с'}")
            print("-" * 85)
            for r in inter_summary:
                print(f"{r['interaction_id']:<5} {r['person_id']:<5} {r['object_id']:<6} "
                      f"{r['object_class']:<12} {r['label']:<22} {r['start_time']:<10} "
                      f"{r['end_time']:<10} {r['duration_sec']}")
        else:
            print(f"--- ВЗАИМОДЕЙСТВИЯ: нет ---")
    print("=" * 80)

    if args.save_report:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report = {
            "video": args.video,
            "processed_at": ts,
            "frames": frame_count,
            "duration_sec": round(elapsed, 1),
            "fps": round(frame_count / elapsed, 1),
            "objects": [
                {"id": tid, "class": oh["class"], "frames": oh["frames"],
                 "first_frame": oh["first"], "last_frame": oh["last"]}
                for tid, oh in sorted(object_history.items())
            ],
            "people": [
                {"id": tid, "frames": ph["frames"], "actions": dict(ph["actions"]),
                 "actions_ru": {translate_action(a): c for a, c in ph["actions"].items()},
                 "first_frame": ph["first"], "last_frame": ph["last"]}
                for tid, ph in sorted(person_history.items())
            ],
            "interactions": interaction_tracker.get_summary() if interaction_tracker else [],
        }
        json_path = out_dir / f"report_{Path(args.video).stem}_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] Отчёт сохранён: {json_path}")

        csv_path = out_dir / f"report_{Path(args.video).stem}_{ts}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["type", "id", "class", "action", "frames", "first_frame", "last_frame", "details"])
            for tid, oh in sorted(object_history.items()):
                w.writerow(["object", tid, translate_object(oh["class"]), "", oh["frames"], oh["first"], oh["last"], ""])
            for tid, ph in sorted(person_history.items()):
                dominant = max(ph["actions"].items(), key=lambda x: x[1])[0] if ph["actions"] else ""
                w.writerow(["person", tid, "", translate_action(dominant), ph["frames"], ph["first"], ph["last"],
                            str({translate_action(a): c for a, c in ph["actions"].items()})])
            if interaction_tracker:
                for r in interaction_tracker.get_summary():
                    w.writerow(["interaction", r["interaction_id"], r["object_class"], r["label"],
                                r["frames"], "", "", f"person={r['person_id']}, object={r['object_id']}"])
        print(f"[OK] CSV сохранён: {csv_path}")


if __name__ == "__main__":
    main()