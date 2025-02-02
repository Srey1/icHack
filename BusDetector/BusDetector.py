import cv2 # Camera information
import numpy as np # preprocess the frames 
# import torch 
from ultralytics import YOLO # Object detection
import pyttsx3 # Text-to-Speech
import easyocr # Detect bus number
import time


def preprocess_for_ocr(image):
    """ Convert image to grayscale and apply thresholding to enhance text. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

model = YOLO("yolov8n.pt") 

reader = easyocr.Reader(["en"]) 

# Change `0` to filepath
cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Bus_Footage_Cropped 20.mp4")
frame_count = 0
bus_boxes = []
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 60 == 0:
        bus_boxes = []
        results = model(frame)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item()) 
                conf = box.conf.item()  
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # (COCO dataset class ID 
                if cls_id == 5 and conf > 0.5:
                    bus_boxes.append((x1, y1, x2, y2))

        for x1, y1, x2, y2 in bus_boxes:           
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"Bus: {conf:.2f}", (x1, y1 - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bus_roi = frame[y1:y2, x1:x2]

            text_results = reader.readtext(bus_roi)

            for (bbox, text, prob) in text_results:
                if prob > 0.5:
                    print(f"Detected Bus Number: {text}")
                    # cv2.putText(frame, text, (x1, y2 + 30), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Bus Detection & Number Recognition", frame)
        # cv2.imshow("LMAO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
cap.release()
cv2.destroyAllWindows()