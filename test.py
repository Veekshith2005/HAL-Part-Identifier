# test_model.py

from ultralytics import YOLO
import cv2

model = YOLO("trained_models/retrained.pt")

img = cv2.imread("hat1.jpg")
results = model(img, conf=0.5)

print("Boxes:", len(results[0].boxes))

for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])

    print(
        "Class:",
        results[0].names[cls],
        "Conf:",
        conf
    )