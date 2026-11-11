#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Общие компоненты: конфиг, трекер, сессия, экспортёры."""

import csv
import io
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SimpleTracker:
    def __init__(self, distance_thresh=100, max_disappear=25):
        self.distance_thresh = distance_thresh
        self.max_disappear = max_disappear
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}

    def update(self, detections):
        for oid in list(self.objects.keys()):
            self.disappeared[oid] += 1
        matched_ids = set()
        now = time.time()
        for (x1, y1, x2, y2, cls, conf) in detections:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best_id = None
            best_dist = float("inf")
            for oid, obj in self.objects.items():
                if obj["class"] != cls:
                    continue
                dist = np.hypot(cx - obj["center"][0], cy - obj["center"][1])
                if dist < self.distance_thresh and dist < best_dist:
                    best_dist = dist
                    best_id = oid
            if best_id is not None and best_id not in matched_ids:
                o = self.objects[best_id]
                o["center"] = (cx, cy)
                o["bbox"] = (int(x1), int(y1), int(x2), int(y2))
                o["last_seen"] = now
                o["frames"] = o.get("frames", 0) + 1
                o["conf"] = float(conf)
                o["active"] = True
                self.disappeared[best_id] = 0
                matched_ids.add(best_id)
            else:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = {
                    "id": oid, "class": cls, "center": (cx, cy),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "first_seen": now, "last_seen": now, "frames": 1,
                    "conf": float(conf), "active": True,
                }
                self.disappeared[oid] = 0
                matched_ids.add(oid)
        for oid in list(self.objects.keys()):
            if self.disappeared[oid] > self.max_disappear:
                self.objects[oid]["active"] = False
            if self.disappeared[oid] > self.max_disappear * 2:
                del self.objects[oid]
                del self.disappeared[oid]
        return self.objects


class Session:
    def __init__(self, target_classes, expected_seconds=None):
        self.target_classes = set(c.lower() for c in target_classes)
        self.expected_seconds = expected_seconds
        self.start_time = time.time()
        self.end_time = None
        self.is_recording = True
        self.targets = {}
        self.ignored = {}

    def update(self, tracker_objects):
        if not self.is_recording:
            return
        now = time.time()
        for oid, obj in tracker_objects.items():
            if not obj.get("active", False):
                continue
            cls = obj["class"].lower()
            if cls in self.target_classes:
                self._touch(self.targets, oid, obj["class"], now)
            else:
                self._touch(self.ignored, oid, obj["class"], now)

    def _touch(self, storage, oid, cls, now):
        if oid not in storage:
            storage[oid] = {
                "id": oid, "class": cls,
                "first_seen": now, "last_seen": now,
                "total_duration": 0.0,
            }
        else:
            rec = storage[oid]
            rec["total_duration"] += now - rec["last_seen"]
            rec["last_seen"] = now

    def stop(self):
        self.is_recording = False
        self.end_time = time.time()
        now = self.end_time
        for storage in (self.targets, self.ignored):
            for rec in storage.values():
                rec["total_duration"] += now - rec["last_seen"]
                rec["last_seen"] = now
        return self._summary()

    def _summary(self):
        def fmt(ts):
            return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        def build_rows(storage, obj_type):
            rows = []
            for rec in storage.values():
                dur = round(rec["total_duration"], 1)
                row = {
                    "object_type": obj_type, "class": rec["class"],
                    "tracker_id": rec["id"],
                    "first_seen": fmt(rec["first_seen"]),
                    "last_seen": fmt(rec["last_seen"]),
                    "total_duration_sec": dur,
                }
                if obj_type == "target" and self.expected_seconds:
                    dev = round(dur - self.expected_seconds, 1)
                    row["expected_sec"] = self.expected_seconds
                    row["deviation_sec"] = dev
                    if abs(dev) <= 5:
                        row["status"] = "норма"
                    elif dev > 0:
                        row["status"] = "превышение"
                    else:
                        row["status"] = "недостаток"
                rows.append(row)
            return rows

        target_rows = build_rows(self.targets, "target")
        ignored_rows = build_rows(self.ignored, "ignored")
        ignored_summary = defaultdict(lambda: {"count": 0, "total_duration_sec": 0.0})
        for r in ignored_rows:
            ign = ignored_summary[r["class"]]
            ign["count"] += 1
            ign["total_duration_sec"] += r["total_duration_sec"]
        ignored_summary = [
            {"class": k, "count": v["count"], "total_duration_sec": round(v["total_duration_sec"], 1)}
            for k, v in ignored_summary.items()
        ]
        return {
            "session_start": fmt(self.start_time), "session_end": fmt(self.end_time),
            "session_duration_sec": round(self.end_time - self.start_time, 1),
            "expected_seconds": self.expected_seconds,
            "target_classes": sorted(list(self.target_classes)),
            "targets": target_rows, "ignored": ignored_rows,
            "ignored_summary": ignored_summary,
        }


class Exporter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = Path(cfg["export"]["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.template = cfg["export"]["filename_template"]

    def _make_filename(self, summary, ext):
        ts = summary["session_end"].replace(":", "-")
        date = datetime.now().strftime("%Y-%m-%d")
        name = self.template.format(timestamp=ts, date=date)
        return self.out_dir / f"{name}.{ext}"

    def to_csv(self, summary):
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "session_start", "session_end", "session_duration_sec",
            "expected_seconds", "object_type", "class", "tracker_id",
            "first_seen", "last_seen", "total_duration_sec",
            "deviation_sec", "status",
        ])
        for r in summary["targets"]:
            writer.writerow([
                summary["session_start"], summary["session_end"],
                summary["session_duration_sec"], summary["expected_seconds"],
                r["object_type"], r["class"], r["tracker_id"],
                r["first_seen"], r["last_seen"], r["total_duration_sec"],
                r.get("deviation_sec", ""), r.get("status", ""),
            ])
        for r in summary["ignored"]:
            writer.writerow([
                summary["session_start"], summary["session_end"],
                summary["session_duration_sec"], summary["expected_seconds"],
                r["object_type"], r["class"], r["tracker_id"],
                r["first_seen"], r["last_seen"], r["total_duration_sec"],
                "", "",
            ])
        for r in summary.get("interactions", []):
            writer.writerow([
                summary["session_start"], summary["session_end"],
                summary["session_duration_sec"], summary["expected_seconds"],
                "interaction", r["label"], r["interaction_id"],
                r["start_time"], r["end_time"], r["duration_sec"],
                "", "",
            ])
        return buf.getvalue().encode("utf-8-sig")

    def to_json(self, summary):
        payload = {
            "meta": {
                "session_start": summary["session_start"],
                "session_end": summary["session_end"],
                "session_duration_sec": summary["session_duration_sec"],
                "expected_seconds": summary["expected_seconds"],
                "target_classes": summary["target_classes"],
            },
            "targets": summary["targets"],
            "ignored": summary["ignored"],
            "ignored_summary": summary["ignored_summary"],
            "interactions": summary.get("interactions", []),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    def to_xlsx(self, summary):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        wb = Workbook()
        ws = wb.active
        ws.title = "Отчёт"
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin"),
        )
        meta = [
            ("Начало сессии", summary["session_start"]),
            ("Окончание сессии", summary["session_end"]),
            ("Длительность сессии, с", summary["session_duration_sec"]),
            ("Норматив, с", summary["expected_seconds"] if summary["expected_seconds"] else "—"),
            ("Целевые классы", ", ".join(summary["target_classes"])),
        ]
        for row_idx, (k, v) in enumerate(meta, 1):
            ws.cell(row=row_idx, column=1, value=k).font = Font(bold=True)
            ws.cell(row=row_idx, column=2, value=v)
        start_row = len(meta) + 2
        headers = ["Тип", "Класс", "ID", "Первое появление", "Последнее появление",
                   "Время в кадре, с", "Норматив, с", "Отклонение, с", "Статус"]
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=start_row, column=col, value=h)
            c.font = header_font; c.fill = header_fill
            c.alignment = Alignment(horizontal="center")
            c.border = thin_border
        row = start_row + 1
        for r in summary["targets"]:
            vals = [
                "целевой", r["class"], r["tracker_id"], r["first_seen"], r["last_seen"],
                r["total_duration_sec"],
                r.get("expected_sec", "—"), r.get("deviation_sec", "—"), r.get("status", "—"),
            ]
            for col, v in enumerate(vals, 1):
                c = ws.cell(row=row, column=col, value=v)
                c.border = thin_border
                if col == 9 and v == "норма":
                    c.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif col == 9 and v in ("превышение", "недостаток"):
                    c.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            row += 1
        start_row = row + 1
        ws.cell(row=start_row, column=1, value="Посторонние объекты").font = Font(bold=True, size=12)
        start_row += 1
        headers2 = ["Класс", "Количество (уникальных ID)", "Суммарное время, с"]
        for col, h in enumerate(headers2, 1):
            c = ws.cell(row=start_row, column=col, value=h)
            c.font = header_font; c.fill = header_fill; c.border = thin_border
        row = start_row + 1
        for r in summary["ignored_summary"]:
            for col, v in enumerate([r["class"], r["count"], r["total_duration_sec"]], 1):
                ws.cell(row=row, column=col, value=v).border = thin_border
            row += 1

        if summary.get("interactions"):
            ws2 = wb.create_sheet("Взаимодействия")
            hi = ["ID", "Человек (ID)", "Предмет (ID)", "Класс предмета", "Операция",
                  "Начало", "Конец", "Длительность, с", "Кадров"]
            for col, h in enumerate(hi, 1):
                c = ws2.cell(row=1, column=col, value=h)
                c.font = header_font; c.fill = header_fill; c.border = thin_border
            for row_idx, r in enumerate(summary["interactions"], 2):
                vals = [r["interaction_id"], r["person_id"], r["object_id"],
                        r["object_class"], r["label"], r["start_time"], r["end_time"],
                        r["duration_sec"], r["frames"]]
                for col, v in enumerate(vals, 1):
                    ws2.cell(row=row_idx, column=col, value=v).border = thin_border

        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column].width = adjusted_width

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

# ═══════════════════════════════════════════════════════
#  Переводы для вывода (русские названия)
# ═══════════════════════════════════════════════════════

RUSSIAN_OBJECTS = {
    "person": "человек",
    "car": "автомобиль",
    "truck": "грузовик",
    "bus": "автобус",
    "train": "поезд",
    "bicycle": "велосипед",
    "motorcycle": "мотоцикл",
    "book": "книга",
    "laptop": "ноутбук",
    "pen": "ручка",
    "pencil": "карандаш",
    "marker": "маркер",
    "headphones": "наушники",
    "earphones": "наушники",
    "watch": "часы",
    "cup": "чашка",
    "mug": "кружка",
    "bottle": "бутылка",
    "cell phone": "телефон",
    "phone": "телефон",
    "smartphone": "смартфон",
    "keyboard": "клавиатура",
    "mouse": "мышь",
    "notebook": "блокнот",
    "scissors": "ножницы",
    "stapler": "степлер",
    # === НОВЫЕ КЛАССЫ (2026-07-01) ===
    "wheel": "колесо",
    "manometer": "манометр",
    "monitor": "монитор",
    "tv": "телевизор",
}

RUSSIAN_ACTIONS = {
    "using_phone": "использует_телефон",
    "walking": "идёт",
    "sitting": "сидит",
    "working": "работает",
    "standing": "стоит",
    "unknown": "неизвестно",
}


def translate_object(cls_name: str) -> str:
    return RUSSIAN_OBJECTS.get(cls_name.lower(), cls_name)


def translate_action(action_name: str) -> str:
    return RUSSIAN_ACTIONS.get(action_name.lower(), action_name)


def put_text_ru(frame, text, pos, font_size=20, color=(255, 255, 255), thickness=1):
    """Рисует русский текст на кадре OpenCV через PIL (поддержка кириллицы)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = None
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    # PIL использует RGB, OpenCV — BGR
    pil_color = color[::-1] if isinstance(color, tuple) else color
    draw.text(pos, text, font=font, fill=pil_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)