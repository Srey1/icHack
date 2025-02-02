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

# * List to store trackers and their previous centers
trackers = []  # Format: [(tracker, prev_center), ...]

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
        last_boxes = bus_boxes
        bus_boxes = []
        results = model(frame)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())  # Get class ID
                conf = box.conf.item()  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                if cls_id == 5 and conf > 0.5:  # Class ID 5 = Bus
                    bus_boxes.append((x1, y1, x2, y2))

                    # * Check if this bus is already being tracked
                    is_new_bus = True
                    for tracker, _ in trackers:
                        tracker_bbox = tracker.get_position()  # Get current tracker position
                        tracker_x1, tracker_y1, tracker_w, tracker_h = map(int, tracker_bbox)
                        tracker_x2, tracker_y2 = tracker_x1 + tracker_w, tracker_y1 + tracker_h

                        # Check if the new bus overlaps with an existing tracker
                        if (x1 < tracker_x2 and x2 > tracker_x1 and
                            y1 < tracker_y2 and y2 > tracker_y1):
                            is_new_bus = False
                            break

                    # * If it's a new bus, initialize a tracker for it
                    if is_new_bus:
                        bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h) format
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        trackers.append((tracker, None))  # Add to trackers list

        # 🚀 Run OCR every 10 YOLO detections to reduce lag
        if ocr_counter % OCR_INTERVAL == 0:
            for i, (x1, y1, x2, y2) in enumerate(bus_boxes):
                lx1, _, lx2, _ = last_boxes[i]
                
                if x1 < lx1 and x2 < lx2:
                    print("moving away")
                else:
                    print("coming closer")
                    
                # Crop the bus number area
                bus_roi = frame[y1:y2, x1:x2]

                # Enhance contrast instead of thresholding
                bus_roi = enhance_contrast(bus_roi)

                # Convert to RGB (PaddleOCR requires RGB)
                bus_roi = cv2.cvtColor(bus_roi, cv2.COLOR_BGR2RGB)

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

    # * Update all trackers and determine directions
    for i, (tracker, prev_center) in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            # * Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # * Calculate center of the bounding box
            center = (x + w // 2, y + h // 2)

            # * Determine direction
            if prev_center is not None:
                if center[0] > prev_center[0]:
                    direction = "Right"
                else:
                    direction = "Left"
                cv2.putText(frame, f"Bus {i+1}: {direction}", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # * Update previous center
            trackers[i] = (tracker, center)  # Update tracker with new center

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