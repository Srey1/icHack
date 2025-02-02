import cv2  # Camera & video processing
import numpy as np  # Image processing
from ultralytics import YOLO  # Object detection
import pyttsx3  # Text-to-Speech (optional)
from paddleocr import PaddleOCR 
import time # OCR for bus number detection

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Initialize PaddleOCR
paddle = PaddleOCR(use_angle_cls=True, lang="en")

# Open video stream (Change `0` for webcam or use file path)
cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Bus_Footage_Cropped 10.mp4")

# Frame tracking
frame_count = 0
bus_boxes = []
ocr_counter = 0  # Track OCR calls
OCR_INTERVAL = 1  # Run OCR every 10 YOLO detections

# 🚀 Function: Enhance contrast instead of thresholding (Preserves text details)
def enhance_contrast(image):
    """ Improve text visibility using CLAHE (Adaptive Histogram Equalization). """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process YOLO detection every 60 frames
    if frame_count % 60 == 0:
        bus_boxes = []
        results = model(frame)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())  # Get class ID
                conf = box.conf.item()  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                if cls_id == 5 and conf > 0.5:  # Class ID 5 = Bus
                    bus_boxes.append((x1, y1, x2, y2))

        # 🚀 Run OCR every 10 YOLO detections to reduce lag
        if ocr_counter % OCR_INTERVAL == 0:
            for x1, y1, x2, y2 in bus_boxes:
                # Crop the bus number area
                bus_roi = frame[y1:y2, x1:x2]

                # Enhance contrast instead of thresholding
                bus_roi = enhance_contrast(bus_roi)

                # Convert to RGB (PaddleOCR requires RGB)
                # bus_roi = cv2.cvtColor(bus_roi, cv2.COLOR_BGR2RGB)
                # cv2.imshow("lMAO", bus_roi)
                # cv2.waitKey(0)
                # Run OCR (Check for None before iterating)
                text_results = paddle.ocr(bus_roi)
                print(text_results)

                if text_results and isinstance(text_results, list):  # Ensure valid OCR results
                    for result in text_results:
                        if result:  # Ensure it's not empty
                            for line in result:
                                if len(line) >= 2:  # Check structure
                                    bbox, (text, confidence) = line  # Extract text & confidence
                                    if confidence > 0.5:  # Adjust threshold
                                        print(f"Detected Bus Number: {text}, Confidence: {confidence}")

        ocr_counter += 1  # Increment OCR tracking counter

    # Show the processed video frame
    cv2.imshow("Bus Detection & Number Recognition", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment frame count
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
cap.release()
cv2.destroyAllWindows()
