import cv2  # Camera & video processing
import numpy as np  # Image processing
from ultralytics import YOLO  # Object detection
from paddleocr import PaddleOCR 
import time # OCR for bus number detection
import re

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Initialize PaddleOCR
paddle = PaddleOCR(use_angle_cls=True, lang="en")

# Open video stream (Change `0` for webcam or use file path)

# daytime
daytime = True
cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Double_Bus_139.mp4") # (3) 139

# night time
# daytime = False
# cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Bus_Footage_Cropped 11.mp4") # (2) 328
# cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Bus_Footage_Cropped 10.mp4") # (1) 360

target_bus_number = "139"

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

def reduce_glow(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust brightness and contrast
    alpha = 0.5  # Contrast control (1.0-3.0)
    beta = -50   # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply median blur to reduce glow
    median = cv2.medianBlur(adjusted, 3)

    # Apply gamma correction
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(median, table)

    # Apply high-pass filter
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    high_pass = cv2.filter2D(gamma_corrected, -1, kernel)
    # high_pass_bgr = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)
    return high_pass
# Variables to store bus number and direction
detected_bus_number = None
detected_direction = None

curr_tracking = False
tracking_delay = 2
curr_tracking_delay = 0
target_bus_prev_coords = None

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip processing if bus number and direction are already detected
    if detected_bus_number is not None and detected_direction == "Left":
        print(f"Bus Number: {detected_bus_number}, Direction: {detected_direction}")
        break  # Exit the loop

    # Process YOLO detection every 60 frames
    if frame_count % 90 == 0 or (curr_tracking and curr_tracking_delay == 0):
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
            for i, (x1, y1, x2, y2) in enumerate(bus_boxes):
                # Crop the bus number area
                bus_roi = frame[y1:y2, x1:x2]

                # Enhance contrast instead of thresholding
                if not daytime:
                    bus_roi = enhance_contrast(bus_roi)
                # cv2.imshow("contrast", bus_roi)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # bus_roi = reduce_glow(bus_roi)

                # Convert to RGB (PaddleOCR requires RGB)
                bus_roi = cv2.cvtColor(bus_roi, cv2.COLOR_BGR2RGB)

                # Run OCR (Check for None before iterating)
                text_results = paddle.ocr(bus_roi)
                # print(text_results)

                if text_results and isinstance(text_results, list):  # Ensure valid OCR results
                    for result in text_results:
                        if result:  # Ensure it's not empty
                            for line in result:
                                if len(line) >= 2:  # Check structure
                                    bbox, (text, confidence) = line  # Extract text & confidence
                                    if confidence > 0.5:  # Adjust threshold
                                        detected_bus_number = re.sub(r"\D", "", text)
                                        print(f"Detected Bus Number: {detected_bus_number}, Confidence: {confidence}")
                                        if detected_bus_number == target_bus_number:
                                            if not curr_tracking:
                                                # start tracking
                                                curr_tracking = True 
                                                curr_tracking_delay = tracking_delay
                                                target_bus_prev_coords = (x1, y1, x2, y2)
                                            elif curr_tracking_delay == 0:
                                                # compare to last known location
                                                prev_x1, prev_y1, prev_x2, prev_y2 = target_bus_prev_coords
                                                prev_center = (prev_x1 + prev_x2) / 2
                                                next_center = (x1 + x2) / 2
                                                if next_center < prev_center:
                                                    detected_direction = "Left"
                                                    print(f"\nTarget bus number {target_bus_number} found moving towards us!\n")
                                                else:
                                                    detected_direction = "Right"
                                                    print(f"\nTarget bus number {target_bus_number} found but its going away from us :(\n")
                                                    curr_tracking = False 
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                cv2.rectangle(frame, (prev_x1, prev_y1), (prev_x2, prev_y2), (255, 0, 0), 2)
                                                cv2.imshow("haha", frame)
                                                cv2.waitKey(0)
                                                cv2.destroyAllWindows()

                                                break
        ocr_counter += 1  # Increment OCR tracking counter
    if curr_tracking:
        curr_tracking_delay -= 1                                    

    # Show the processed video frame
    cv2.imshow("Bus Detection & Number Recognition", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment frame count

end_time = time.time()
elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time}")
cap.release()
cv2.destroyAllWindows()

# Output the final result
if detected_bus_number == target_bus_number and detected_direction == "Left":
    print(f"\nTarget found! - Bus Number: {detected_bus_number}, approaching!\n")
else:
    print("\nNo matching bus number and direction detected.\n")