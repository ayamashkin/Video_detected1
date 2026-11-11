# ═══════════════════════════════════════════════════════
#  Руководство: как поднять FPS с 5-8 до 25-30+
#  2026-07-01 13:05:00 MSK
# ═══════════════════════════════════════════════════════

## Проблема

Видео с низким FPS (5-8 кадров/сек) — это ограничение **самого видеофайла**.
Если обработка идёт медленнее, чем FPS видео — значит инференс тормозит.

## Диагностика

```bash
# Узнать FPS видео
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1 data/video.mp4

# Или через Python
python -c "import cv2; cap=cv2.VideoCapture('data/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS))"
```

## Решения (по приоритету)

### 1. GPU (самый эффективный) — +10× скорости

```yaml
inference:
  device: "0"       # CUDA GPU #0
  half: true        # FP16 — ещё +30% на GPU
```

Требования: NVIDIA GPU с CUDA ≥ 11.8, PyTorch с CUDA.
```bash
# Проверка
python -c "import torch; print(torch.cuda.is_available())"
# Должно вывести: True
```

### 2. Уменьшение imgsz — +2-3× скорости

```yaml
inference:
  imgsz: 320        # вместо 640
  # 640 → 320 = ~4× меньше пикселей = ~2-3× быстрее
```

⚠️ Мелкие предметы (< 50×50 px на кадре) могут пропасть.

### 3. skip_frames — уже работает, но при низком FPS видео ставьте 1

```yaml
inference:
  skip_frames: 1    # обрабатывать ВСЕ кадры (видео и так 5-8 FPS)
```

При `skip_frames: 2` на видео 5 FPS обрабатывается 2.5 кадра/с — это тормозит.

### 4. TensorRT / ONNX — +50-100% на GPU

```python
# Экспорт в TensorRT (требует NVIDIA GPU)
from ultralytics import YOLO
model = YOLO("yolov8s-worldv2.pt")
model.export(format="engine", imgsz=480, half=True, device=0)
# Получите yolov8s-worldv2.engine
```

```yaml
# Использование в config.yaml
detection:
  model: "models/yolov8s-worldv2.engine"
```

### 5. Batch inference для файлов (analyze.py) — +2×

Вместо обработки кадра за кадром — обрабатывать батч из 4-8 кадров.

```python
# В analyze.py заменить цикл на:
batch = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    batch.append(frame)
    if len(batch) == 4:
        results = model(batch, stream=True)  # batch inference
        for r in results:
            process(r)
        batch = []
```

### 6. Многопоточность (для webcam / RTSP)

```python
# В app.py добавить буферизированный захват кадров
import queue
import threading

frame_queue = queue.Queue(maxsize=2)

def capture_thread():
    while running:
        ok, frame = cap.read()
        if ok:
            if frame_queue.full():
                frame_queue.get()  # выбросить старый
            frame_queue.put(frame)

threading.Thread(target=capture_thread, daemon=True).start()

# В основном цикле:
frame = frame_queue.get()  # не блокирует инференс
```

### 7. Уменьшение разрешения видео (препроцессинг)

```bash
# Пережать видео до 640×360 перед анализом
ffmpeg -i input.mp4 -vf "scale=640:360" -r 15 -c:v libx264 -preset fast output.mp4
```

## Таблица: ожидаемый FPS

| Конфигурация | CPU (i7) | GPU (RTX 3060) |
|---|---|---|
| yolov8s-worldv2.pt, imgsz=640 | 3-5 FPS | 40-50 FPS |
| yolov8s-worldv2.pt, imgsz=320 | 8-12 FPS | 80-100 FPS |
| yolov8s-worldv2.engine, imgsz=320, half | — | 120-150 FPS |
| + skip_frames: 2 | 16-24 FPS | 200+ FPS |

## Рекомендуемый config.yaml для видео 5-8 FPS

```yaml
detection:
  model: "models/yolov8s-worldv2.pt"
  confidence_threshold: 0.35
  iou_threshold: 0.45

inference:
  device: "0"       # GPU если есть, иначе "cpu"
  imgsz: 480        # компромисс: не 320 (мелкие), не 640 (медленно)
  half: true        # только для GPU!
  skip_frames: 1    # видео и так медленное — обрабатываем все кадры
```

## Скрипт бенчмарка

```bash
python benchmark.py --video data/MOV_20260701_122113_854.mp4 --model models/yolov8s-worldv2.pt --imgsz 480 --device 0

python benchmark.py --video data/MOV_20260701_122113_854.mp4 --model models/yolov8s-worldv2.pt --imgsz 480

```
