# ═══════════════════════════════════════════════════════
#  Быстрый старт: CWD1500 на CPU
#  2026-07-01 14:30:00 MSK
# ═══════════════════════════════════════════════════════

## Структура ваших данных

train/
  CWD1500/
    CWD1500_train/
      obj.names
      obj_train_data/     # .jpg + .txt
    CWD1500_val/
      obj.names
      obj_train_data/     # .jpg + .txt
    CWD1500_test/
      obj.names
      obj_train_data/     # .jpg + .txt

## Шаг 1. Конвертация (1 минута)

```bash
cd E:\Video_detected1

# Скопируйте convert_cwd1500.py в папку проекта
# Запустите:
python convert_cwd1500.py --input train/CWD1500 --output dataset_cwd1500
```

Результат:
```
dataset_cwd1500/
  images/
    train/   # .jpg из CWD1500_train/obj_train_data/
    val/     # .jpg из CWD1500_val/obj_train_data/
    test/    # .jpg из CWD1500_test/obj_train_data/
  labels/
    train/   # .txt (YOLO format)
    val/
    test/
  data.yaml
```

## Шаг 2. Обучение на CPU

```bash
# БЫСТРАЯ ПРОВЕРКА (20 эпох, imgsz=320) — ~2-4 часа
python train_cpu.py --data dataset_cwd1500/data.yaml --epochs 20 --imgsz 320 --batch 4

# ПОЛНОЕ ОБУЧЕНИЕ (50 эпох, imgsz=416) — ~6-12 часов
python train_cpu.py --data dataset_cwd1500/data.yaml --epochs 50 --imgsz 416 --batch 4

# С ЗАМОРОЗКОЙ BACKBONE (ещё быстрее, transfer learning)
python train_cpu.py --data dataset_cwd1500/data.yaml --epochs 50 --imgsz 416 --batch 4 --freeze 10

# ПРОДОЛЖИТЬ ПРЕРВАННОЕ (если закрылось окно/свет выключился)
python train_cpu.py --data dataset_cwd1500/data.yaml --epochs 50 --imgsz 416 --batch 4 --resume
```

## Шаг 3. Использование в проекте

```bash
# Лучший вес лежит тут:
runs/detect/cwd1500_cpu/weights/best.pt

# Копируем в проект
copy runs\detect\cwd1500_cpu\weights\best.pt models\best.pt
```

В `config.yaml`:
```yaml
detection:
  model: "models/best.pt"
  confidence_threshold: 0.35

classes:
  mode: "coco"   # классы встроены в .pt (car, wheel)
```

## ⚠️ Важно

1. **Обучение на CPU медленное.** Не закрывайте окно — каждая эпоха сохраняется.
2. **Если прервалось** — запустите с `--resume`, продолжится с последней эпохи.
3. **CWD1500 содержит только `car` и `wheel`.** Для `pressure gauge`, `monitor`, `laptop` нужны другие датасеты или используйте YOLO-World zero-shot.
4. **Рекомендуемый план:**
   - Сначала 20 эпох для проверки (mAP должен быть > 0.5)
   - Если всё ок — 50 эпох для финальной модели


```bash

python merge_datasets.py --datasets "cwd:dataset_cwd1500/data.yaml:wheel,car" "ls:dataset_labelstudio/data.yaml:car,manometer,person,wheel" --output dataset_merged
    ```