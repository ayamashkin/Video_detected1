#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ManometerCapture — сохранение крупных планов манометра с фильтром по размеру.

Фильтр: сохраняет только если bbox манометра занимает >= min_size_ratio площади кадра.
Это гарантирует, что сохраняются только крупные планы (пригодные для чтения).
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class ManometerCapture:
    """
    'Фотографирует' манометр при детекции, но только если он достаточно крупный.
    """

    def __init__(self, output_dir="./output/manometer_crops",
                 save_interval=5, min_size_ratio=0.05):
        """
        Args:
            output_dir: куда сохранять
            save_interval: минимум кадров между сохранениями одного ID
            min_size_ratio: минимальная доля площади кадра (0.05 = 5%%)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.min_size_ratio = min_size_ratio
        self._frame_counter = 0
        self._last_saved = {}
        self._skipped_count = 0  # статистика
        self._saved_count = 0

    def _bbox_ratio(self, bbox, frame_shape):
        """Вычислить долю площади bbox от площади кадра."""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        return bbox_area / frame_area if frame_area > 0 else 0

    def save(self, frame, bbox, tracker_id=None, frame_id=0, label="manometer"):
        """
        Сохраняет крупный план манометра, если он достаточно большой.
        Returns: (crop_path, debug_text)
        """
        self._frame_counter += 1

        # Проверка размера
        ratio = self._bbox_ratio(bbox, frame.shape)
        if ratio < self.min_size_ratio:
            self._skipped_count += 1
            return None, f"too_small:{ratio:.3f}<{self.min_size_ratio}"

        # Пропуск по интервалу
        if tracker_id in self._last_saved:
            if (self._frame_counter - self._last_saved[tracker_id]) < self.save_interval:
                return None, "skipped_by_interval"

        # Crop с padding
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad = 0.15
        px, py = int(bw * pad), int(bh * pad)
        x1p = max(0, x1 - px)
        y1p = max(0, y1 - py)
        x2p = min(w, x2 + px)
        y2p = min(h, y2 + py)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return None, "empty_crop"

        # Сохранение
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tid = tracker_id or "x"
        fname = f"{label}_f{frame_id:06d}_id{tid}_sz{ratio:.3f}_{ts}.jpg"
        path = self.output_dir / fname
        cv2.imwrite(str(path), crop)

        self._last_saved[tracker_id] = self._frame_counter
        self._saved_count += 1
        return path, f"saved:{path.name} ratio={ratio:.3f}"

    def draw_label(self, frame, bbox, text="MANO"):
        """Рисует метку на кадре."""
        x1, y1, x2, y2 = bbox
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = x1, y1 - 5
        cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def get_stats(self):
        """Возвращает статистику сохранения."""
        return {
            "saved": self._saved_count,
            "skipped": self._skipped_count,
            "output_dir": str(self.output_dir),
        }