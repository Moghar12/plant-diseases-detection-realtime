import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('best.pt')

image_path = 'img.jpg'
image = cv2.imread(image_path)

results = model(image)

diseases_detected = False
for result in results:
    if len(result.boxes) > 0:
        diseases_detected = True
        for box in result.boxes:
            cls = box.cls
            label = result.names[int(cls)]
            cv2.putText(image, label, (int(box.xyxy[0]), int(box.xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    image = result.plot()

if not diseases_detected:
    cv2.putText(image, "Normal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 3)

# Display the image with detections
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Plant Disease Detection')
plt.axis('off')  # Hide axis
plt.show()
