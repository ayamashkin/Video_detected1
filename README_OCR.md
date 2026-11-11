# OCR манометра — инструкция по установке

## Что делает

Когда детектор находит **манометр** (`manometer`):
1. Вырезает крупный план циферблата
2. Увеличивает и предобрабатывает (контраст, бинаризация)
3. Запускает OCR (Tesseract) — считывает цифры
4. Сохраняет крупный план в `output/manometer_crops/`
5. Отображает значение на видео

## Установка Tesseract OCR

### Windows

1. Скачать установщик:
   https://github.com/UB-Mannheim/tesseract/wiki
   (берите `tesseract-ocr-w64-setup-5.x.x.exe`)

2. Установить в `C:\Program Files\Tesseract-OCR`

3. Добавить в PATH:
   ```
   Панель управления → Система → Дополнительные параметры системы →
   Переменные среды → Path → Добавить:
   C:\Program Files\Tesseract-OCR
   ```

4. Перезапустить CMD / PowerShell / PyCharm

5. Проверить:
   ```bash
   tesseract --version
   ```
   Должно вывести версию (например, `tesseract 5.3.0`).

6. Установить Python-обёртку:
   ```bash
   pip install pytesseract
   ```

### Linux

```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

### macOS

```bash
brew install tesseract
pip install pytesseract
```

## Запуск

```bash
# 1. Скопируйте файлы
python action_recognition_with_ocr.py --video data/video.mp4 --no-display
```

## Настройка в config.yaml

```yaml
manometer_ocr:
  enabled: true
  output_dir: "./output/manometer_crops"
  save_crops: true
  ocr_interval: 5          # каждые 5 кадров (для скорости)
```

| Параметр | Описание |
|----------|----------|
| `enabled` | Включить/выключить OCR |
| `output_dir` | Куда сохранять крупные планы |
| `save_crops` | Сохранять обработанные изображения |
| `ocr_interval` | Запускать OCR каждые N кадров (1 = каждый кадр, медленно) |

## Производительность на CPU

| ocr_interval | Эффект |
|--------------|--------|
| 1 | Каждый кадр — точнее, но медленнее (~+0.3 с/кадр) |
| 5 | Рекомендуется — баланс скорость/точность |
| 10 | Максимальная скорость, но значения реже обновляются |

## Формат сохранённых кропов

```
output/manometer_crops/
  mano_f000123_id1001_20260713_143052.png
  mano_f000128_id1001_20260713_143057.png
```

Имя: `mano_f{frame_id:06d}_id{tracker_id}_{timestamp}.png`

## Ограничения

- Требуется **чёткий циферблат** (размытые/далёкие манометры не считаются)
- Цифры должны быть **контрастными** (тёмные на светлом или наоборот)
- OCR может путать `0` и `O`, `1` и `l` — применяется фильтрация только цифр
- Если манометр **повёрнут** — может не распознаться (нужен deskew)
