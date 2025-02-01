import cv2 # Camera information
import numpy as np # preprocess the frames 
# import torch 
from ultralytics import YOLO # Object detection
import pyttsx3 # Text-to-Speech
import easyocr # Detect bus number

model = YOLO("yolov8n.pt") 

reader = easyocr.Reader(["en"]) 

# Change `0` to filepath
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item()) 
            conf = box.conf.item()  
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # (COCO dataset class ID 5)
            if cls_id == 5 and conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bus: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                bus_roi = frame[y1:y2, x1:x2]

                text_results = reader.readtext(bus_roi)

                for (bbox, text, prob) in text_results:
                    if prob > 0.5:
                        print(f"Detected Bus Number: {text}")
                        cv2.putText(frame, text, (x1, y2 + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Bus Detection & Number Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()