# Видеодетектор с контролем сессий

Веб-приложение + консольный инструмент для измерения времени нахождения объектов в кадре.
Поддерживает **стандартные COCO-классы** и **произвольные объекты** через YOLO-World (zero-shot).

---

## Структура проекта

```
video-detector/
├── app.py              # Flask + веб-интерфейс (импортирует из core)
├── analyze.py          # Консольный анализатор файла (импортирует из core)
├── core.py             # Общие классы: load_config, SimpleTracker, Session, Exporter
├── config.yaml         # Вся конфигурация
├── requirements.txt    # Зависимости
├── output/             # Автосохранение отчётов
└── templates/
    └── index.html      # Веб-интерфейс
```

---

## Быстрый старт

```bash
pip install -r requirements.txt
pip install -r requirements_extra.txt

# Веб-интерфейс (камера / файл / RTSP)
python app.py

# Полный лог в конце (всегда)
python analyze.py --video "data/17. Задание 2, часть 8.mp4" --expected 300

# + логирование новых объектов в реальном времени (verbose)
python analyze.py --video "data/17. Задание 2, часть 8.mp4" --verbose --save-report

# детектирование всех объектов из coco
python analyze.py --video "data/17. Задание 2, часть 8.mp4" --audit
```

---

## Ускорение обработки (6 FPS → 30+ FPS)

Рычаги в `config.yaml` → секция `inference`:

```yaml
inference:
  device: null        # "0" = GPU, "cpu" = CPU, null = авто
  imgsz: 480        # 640=стандарт, 480=компромисс, 320=макс. скорость
  half: false       # true = FP16 (ускоряет на GPU)
  skip_frames: 2    # 1=все кадры, 2=каждый 2-й (2× скорость), 3=каждый 3-й
```

| Параметр | Значение | Эффект |
|---|---|---|
| `skip_frames: 2` | Анализ каждый 2-й кадр | **+100%** скорости, промежуточные кадры используют старые рамки (маркер `[i]`) |
| `imgsz: 320` | Вход 320×320 | **+300%** скорости, мелкие предметы могут пропасть |
| `device: "0"` | CUDA GPU | **+1000%** скорости (если есть видеокарта) |
| `half: true` | FP16 | **+30%** на GPU |

---

## Zero-shot (произвольные объекты: ручки, наушники, часы)

```yaml
detection:
  model: "yolov8s-worldv2.pt"   # YOLO-World вместо стандартной модели

classes:
  mode: "custom"
  custom:
    - watch
    - cup
    - pen
    - headphones
    - laptop

targets:
  classes:
    - pen
    - headphones
    - cup
```

> Классы задаются на **английском**. YOLO-World обучен на англоязычных prompt'ах.

---

## Конфигурация (`config.yaml`)

```yaml
detection:
  source: 0                    # 0=webcam, "path"=файл, "rtsp://..."
  model: "yolov8n.pt"          # или "yolov8s-worldv2.pt" для zero-shot
  confidence_threshold: 0.4
  iou_threshold: 0.45
  buffer_size: 1

inference:
  device: null
  imgsz: 640
  half: false
  skip_frames: 1

classes:
  mode: "coco"                 # "coco" или "custom"
  custom: []

targets:
  classes: [car, truck, person]
  expected_seconds: null       # норматив, сек

tracking:
  tracker: "bytetrack.yaml"    # bytetrack.yaml | botsort.yaml | null
  distance_threshold: 120
  max_disappear: 30

export:
  output_dir: "./output"
  formats: [csv, xlsx, json]
  filename_template: "session_{date}_{timestamp}"

ui:
  refresh_interval_ms: 500
```

---

## Форматы экспорта

| Формат | Содержимое | Применение |
|---|---|---|
| **CSV** | Плоская таблица, UTF-8 с BOM | pandas, 1С |
| **Excel (.xlsx)** | Мета + таблицы, цветовая индикация | Отчёт для руководства |
| **JSON** | Структурированные данные | Интеграция с АСУ |

---

## API Endpoints (app.py)

| Метод | URL | Описание |
|---|---|---|
| `GET`  | `/` | Веб-интерфейс |
| `GET`  | `/video_feed` | MJPEG-поток |
| `POST` | `/api/start` | Старт сессии |
| `POST` | `/api/stop` | Стоп сессии, JSON-отчёт |
| `GET`  | `/api/status` | Статус |
| `GET`  | `/api/export/csv` | Скачать CSV |
| `GET`  | `/api/export/xlsx` | Скачать Excel |
| `GET`  | `/api/export/json` | Скачать JSON |
| `GET`  | `/api/config` | Публичный конфиг |

---

## Типовые сценарии

### Контроль рабочего места (ручка, наушники, часы)

```yaml
detection:
  model: "yolov8s-worldv2.pt"
classes:
  mode: "custom"
  custom:
    - pen
    - headphones
    - watch
    - cup
targets:
  classes:
    - pen
    - headphones
  expected_seconds: 300
inference:
  skip_frames: 2
  imgsz: 480
```

### Проверка нарушений (посторонние в зоне)

```yaml
targets:
  classes: [person]
  expected_seconds: null
```

---

## Расширение

- **Ещё больше классов:** добавьте строки в `classes.custom` (до ~1000 prompt'ов).
- **Точность:** если `yolov8s-worldv2.pt` путает похожие объекты — используйте `yolov8l-worldv2.pt`.
- **Множество файлов:** оберните `analyze.py` в `for` цикл (bash / PowerShell).
- **Сохранение видео:** флаг `--save-video` в `analyze.py` создаёт `annotated_*.mp4` с рамками.
- **Уведомления:** при статусе `превышение` отправляйте HTTP-запрос или пишите в Telegram.


---

## Распознавание действий (Action Recognition)

Для задач контроля операций персонала (например: "сборка", "сварка", "работа за столом", "использование телефона") детекции объектов недостаточно — нужно понимать **что делает человек**.

### Подходы

| Подход | Модели | Скорость | Точность | Применение |
|---|---|---|---|---|
| **Skeleton + LSTM** | YOLOv8-pose + custom classifier | 30+ FPS (GPU) | Средняя | Real-time контроль рабочих операций |
| **RGB 3D-CNN** | SlowFast, X3D (MMAction2) | 5–15 FPS | Высокая | Offline анализ архивов, аудит |
| **Two-stage** | YOLO (детекция) → SlowFast (классификация crop) | 10–20 FPS | Очень высокая | Смешанный режим |
| **VLM (Zero-shot)** | Video-LLaMA, LLaVA-Video | <1 FPS | Высокая (описание) | Генерация текстовых отчётов |

### Рекомендация для внедрения

**Вариант А (Real-time, легко дообучить)** — файл `action_recognition.py`:
```bash
pip install ultralytics

python action_recognition.py --video "data/rutube-default-1468-0.ts" --window 30
python action_recognition.py --video "data/17. Задание 2, часть 8.mp4" --window 30
python action_recognition.py --video "data/MOV_20260701_122132_135.mp4" --window 20
python action_recognition.py --video "data/MOV_20260701_122117_483.mp4" --window 30
python action_recognition.py --video "data/MOV_20260701_122113_854.mp4" --window 30

python action_recognition_dual.py --video data/MOV_20260701_122132_135.mp4 --coco-model models/yolov8n.pt --custom-model models/best.pt --custom-classes wheel --window 30
python action_recognition_dual.py --video data/MOV_20260701_122117_483.mp4 --coco-model models/yolov8n.pt --custom-model models/best.pt --custom-classes wheel --window 30
python action_recognition_dual.py --video data/MOV_20260701_122113_854.mp4 --coco-model models/yolov8n.pt --custom-model models/best.pt --custom-classes wheel --window 30

python action_recognition_dual.py --video "data/17. Задание 2, часть 8.mp4" --coco-model models/yolov8n.pt --custom-model models/best.pt --custom-classes wheel --window 30

python action_recognition_with_ocr.py --video "data/MOV_20260701_122117_483.mp4" --window 30
python action_recognition_with_ocr.py --video "data/MOV_20260701_122132_135.mp4" --window 30
python action_recognition_with_ocr.py --video "data/MOV_20260701_122113_854.mp4" --window 30


python action_recognition_with_ocr.py --video "data/video1.mp4" --window 30
python action_recognition_with_ocr.py --video "data/260901 ПАК ЛАБОРАТОРИЯ MPEG4.mp4" --window 30

python action_recognition_with_ocr.py --video "data/261020 ВИДЕО ИСПЫТАНИЯ КВП.mp4" --window 30



```

Архитектура:
1. **YOLOv8-pose** (`yolov8n-pose.pt`, 6 MB) — детекция 17 keypoints человека
2. **ActionBuffer** — накапливает 30 кадров (1 сек @ 30 FPS) нормализованных keypoints
3. **Классификатор** — в демо используется эвристика; в production заменяется на обученную LSTM/Transformer

Доступные действия (демо-эвристика): `standing`, `sitting`, `walking`, `working`, `using_phone`.

**Вариант Б (Высокая точность, offline)** — MMAction2:
```bash
pip install openmim

# 1. Обновите setuptools, pip и wheel
python -m pip install --upgrade pip setuptools wheel

# 2. Если mim всё ещё не работает — обновите и его
python -m pip install --upgrade openmim


mim install mmaction2
```

```python
from mmaction.apis import init_recognizer, inference_recognizer

config = "configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py"
checkpoint = "slowfast_r50_256e_kinetics400_rgb_20210728-4ab16c62.pth"
model = init_recognizer(config, checkpoint, device="cuda:0")
results = inference_recognizer(model, "video.mp4")
# results: [("archery", 0.92), ("bowling", 0.03), ...]
```

### Интеграция в существующий VideoProcessor

```python
from action_recognition import ActionRecognizer

class VideoProcessor:
    def __init__(self, cfg):
        ...
        self.action_recognizer = ActionRecognizer(
            pose_model="yolov8n-pose.pt",
            window_size=30,
            device=self.device,
        )

    def _loop(self):
        ...
        # после трекинга объектов
        actions = self.action_recognizer.update(frame, tracker_objects)
        for tid, (action, conf) in actions.items():
            if tid in tracker_objects:
                tracker_objects[tid]["action"] = action
                tracker_objects[tid]["action_conf"] = conf
        ...
```

### Дообучение под свои операции

Для корпоративных сценариев (ОПК, производство) соберите 50–100 видео каждого действия и обучите простой классификатор:

```python
import torch.nn as nn

class ActionLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: (batch, seq=30, feat=34)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

Классы для производства:
- `assembly` — сборка
- `welding` — сварка
- `inspection` — контроль
- `idle` — простой
- `safety_violation` — нарушение (например, отсутствие каски)


---

## Human-Object Interaction (HOI) — совмещение действий и предметов

Система определяет **не только объекты и людей, но и взаимодействия между ними**:  
*«Человек #3 + ручка = письмо/разметка (12.5 сек)»*.

### Архитектура

```
Кадр
  │
  ├──► YOLOv8 (объекты) ──► ByteTrack ──► objects {id, bbox, class}
  │
  └──► YOLOv8-pose (люди) ──► ByteTrack ──► poses {id, bbox, keypoints (17,2)}
           │
           └──► InteractionTracker ──► proximity (wrist → object center)
                    │
                    └──► Правило: action + object_class → label
                             └──► "письмо/разметка" | "работа за ПК" | "перерыв/питьё"
```

### Включение в config.yaml

```yaml
interactions:
  enabled: true
  pose_model: "yolov8n-pose.pt"   # ~6 MB, скачивается автоматически
  proximity_threshold: 150         # пиксели от запястья до центра предмета
  min_interaction_frames: 5        # минимум кадров для подтверждения
  max_gap_seconds: 1.0             # допустимый разрыв (рука ушла на 1 сек)

  rules:
    - action: "any"                # "any" = любое действие человека
      objects: ["pen", "pencil"]
      label: "письмо/разметка"
    - action: "any"
      objects: ["laptop", "keyboard"]
      label: "работа за ПК"
    - action: "any"
      objects: ["cup", "mug", "bottle"]
      label: "перерыв/питьё"
```

### Как это работает

1. **YOLOv8-pose** детектирует человека и 17 keypoints (скелет)
2. **InteractionTracker** вычисляет расстояние от `left_wrist` (9) и `right_wrist` (10) до центра bbox каждого предмета
3. Если расстояние < `proximity_threshold` **и** предмет входит в список `objects` правила — фиксируется взаимодействие
4. Взаимодействие длится, пока рука рядом. Если рука ушла более чем на `max_gap_seconds` — запись завершается

### Отчёт

В Excel добавляется лист **«Взаимодействия»**:

| ID | Человек (ID) | Предмет (ID) | Класс предмета | Операция | Начало | Конец | Длительность, с | Кадров |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 7 | pen | письмо/разметка | 14:32:10 | 14:32:22 | 12.5 | 15 |
| 1 | 3 | 12 | cup | перерыв/питьё | 14:35:01 | 14:35:45 | 44.0 | 52 |

### Производительность

| Режим | Модели | CPU (i7) | GPU (RTX 3060) |
|---|---|---|---|
| Только объекты | yolov8n.pt | 25 FPS | 120 FPS |
| + Pose (HOI) | yolov8n.pt + yolov8n-pose.pt | 10–12 FPS | 80 FPS |
| + Skip frames=2 | обе модели | 20 FPS | 140 FPS |

> Совет: для CPU используйте `skip_frames: 2` в `inference`, чтобы pose запускалась через кадр.

### Интеграция с Action Recognition

Если подключён `action_recognition.py` (LSTM на скелете), поле `action` из `tracker_objects` передаётся в `InteractionTracker`. Тогда правила можно уточнить:

```yaml
rules:
  - action: "working"            # только если человек в позе "работа"
    objects: ["pen"]
    label: "письмо/разметка"
  - action: "using_phone"
    objects: ["cell phone"]
    label: "использование телефона (нарушение)"
```

### Пример запуска

```bash
# Включить HOI в config.yaml (interactions.enabled: true)
python analyze.py --video "data/17. Задание 2, часть 8.mp4" --expected 300
python analyze.py --video "data/rutube-default-1468-0.ts" --expected 300

# Результат в output/session_*.xlsx — лист "Взаимодействия"
```
