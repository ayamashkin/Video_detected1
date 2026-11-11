#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Human-Object Interaction (HOI) tracker — proximity-based."""

import time
import numpy as np
from collections import defaultdict


class InteractionTracker:
    def __init__(self, cfg):
        icfg = cfg.get("interactions", {})
        self.enabled = icfg.get("enabled", False)
        self.rules = icfg.get("rules", [])
        self.proximity = icfg.get("proximity_threshold", 150)
        self.min_frames = icfg.get("min_interaction_frames", 5)
        self.max_gap_sec = icfg.get("max_gap_seconds", 1.0)
        self.active = {}
        self.history = []
        self._next_id = 0

    def _match_rule(self, action_hint, obj_class):
        obj_class = obj_class.lower()
        action_hint = (action_hint or "unknown").lower()
        for rule in self.rules:
            rule_action = rule.get("action", "").lower()
            if rule_action and rule_action != action_hint and rule_action != "any":
                continue
            if obj_class in [o.lower() for o in rule.get("objects", [])]:
                return rule.get("label", f"{action_hint}+{obj_class}")
        return None

    @staticmethod
    def _wrist_distance(keypoints, bbox):
        if keypoints is None or len(keypoints) < 17:
            return float("inf")
        wrists = [keypoints[9], keypoints[10]]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        min_dist = float("inf")
        for w in wrists:
            if w[0] == 0 and w[1] == 0:
                continue
            d = np.hypot(w[0] - cx, w[1] - cy)
            if d < min_dist:
                min_dist = d
        return min_dist

    def update(self, poses, objects, frame_shape=None):
        if not self.enabled:
            return []
        now = time.time()
        current_pairs = set()
        for pid, pdata in poses.items():
            kpts = pdata.get("keypoints")
            if kpts is None:
                continue
            for oid, odata in objects.items():
                if not odata.get("active", False):
                    continue
                dist = self._wrist_distance(kpts, odata["bbox"])
                if dist > self.proximity:
                    continue
                action_hint = odata.get("action", "unknown")
                label = self._match_rule(action_hint, odata["class"])
                if label is None:
                    continue
                key = (pid, oid)
                current_pairs.add(key)
                if key not in self.active:
                    self.active[key] = {
                        "interaction_id": self._next_id,
                        "person_id": pid, "object_id": oid,
                        "object_class": odata["class"],
                        "label": label,
                        "start_time": now, "last_time": now, "frames": 1,
                    }
                    self._next_id += 1
                else:
                    self.active[key]["last_time"] = now
                    self.active[key]["frames"] += 1

        # Завершаем пропавшие взаимодействия
        for key in list(self.active.keys()):
            if key not in current_pairs:
                rec = self.active[key]
                if (now - rec["last_time"]) > self.max_gap_sec:
                    rec = self.active.pop(key)
                    if rec["frames"] >= self.min_frames:
                        rec["end_time"] = now
                        self.history.append(rec)

        # Формируем список активных — с проверкой наличия ID
        active_list = []
        for key, rec in self.active.items():
            if rec["frames"] >= self.min_frames:
                # FIX: проверяем, что person и object всё ещё в кадре
                if rec["person_id"] not in poses or rec["object_id"] not in objects:
                    continue
                active_list.append({
                    "interaction_id": rec["interaction_id"],
                    "person_id": rec["person_id"],
                    "object_id": rec["object_id"],
                    "object_class": rec["object_class"],
                    "label": rec["label"],
                    "duration_sec": round(now - rec["start_time"], 1),
                    "bbox_person": poses[rec["person_id"]]["bbox"],
                    "bbox_object": objects[rec["object_id"]]["bbox"],
                })
        return active_list

    def get_summary(self):
        now = time.time()
        for key in list(self.active.keys()):
            rec = self.active.pop(key)
            if rec["frames"] >= self.min_frames:
                rec["end_time"] = rec.get("end_time", now)
                self.history.append(rec)
        rows = []
        for rec in self.history:
            end = rec.get("end_time", now)
            rows.append({
                "interaction_id": rec["interaction_id"],
                "person_id": rec["person_id"],
                "object_id": rec["object_id"],
                "object_class": rec["object_class"],
                "label": rec["label"],
                "start_time": time.strftime("%H:%M:%S", time.localtime(rec["start_time"])),
                "end_time": time.strftime("%H:%M:%S", time.localtime(end)),
                "duration_sec": round(end - rec["start_time"], 1),
                "frames": rec["frames"],
            })
        return rows

    def draw(self, frame, interactions):
        import cv2
        for inter in interactions:
            px1, py1, px2, py2 = inter["bbox_person"]
            ox1, oy1, ox2, oy2 = inter["bbox_object"]
            pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
            ocx, ocy = int((ox1 + ox2) / 2), int((oy1 + oy2) / 2)
            cv2.line(frame, (pcx, pcy), (ocx, ocy), (255, 0, 255), 2)
            label = f"{inter['label']} ({inter['duration_sec']}s)"
            mx, my = (pcx + ocx) // 2, (pcy + ocy) // 2
            cv2.putText(frame, label, (mx, max(my - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return frame