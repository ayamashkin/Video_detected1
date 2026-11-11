import cv2
from manometer_reader import AnalogManometerReader

reader = AnalogManometerReader(
    min_value=0, max_value=10,
    zero_angle=90, max_angle=270,
    output_dir="./output/test_mano",
    save_crops=True,
    process_interval=1
)

frame = cv2.imread("raw_labelstudio/images/IMG_20260702_103427.jpg")
# bbox примерный: [x1, y1, x2, y2]
bbox = [100, 100, 400, 400]  # подберите под ваше фото!

value, path, debug = reader.process(frame, bbox, tracker_id=1, frame_id=1)
print(f"value={value}, path={path}, debug={debug}")