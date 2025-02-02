import cv2 # Camera information
import numpy as np # preprocess the frames 
# import torch 
from ultralytics import YOLO # Object detection
import pyttsx3 # Text-to-Speech
import easyocr # Detect bus number
import time


def preprocess_for_ocr(image):
    """ Reduce LED glow and enhance contrast for better OCR accuracy. """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations (Top-Hat filtering) to remove background glow
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Small kernel for LED lights
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Apply bilateral filter to further smooth out glow
    filtered = cv2.bilateralFilter(tophat, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply adaptive thresholding (helps extract LED numbers better)
    adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 10)

    # Apply CLAHE to boost contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(adaptive_thresh)

    # Check brightness level & invert if needed
    if np.mean(contrast_enhanced) > 127:  # If too bright, invert
        contrast_enhanced = cv2.bitwise_not(contrast_enhanced)

    return contrast_enhanced


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

    return high_pass


model = YOLO("yolov8n.pt") 

reader = easyocr.Reader(["en"]) 

# Change `0` to filepath
cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Bus_Footage_Cropped 10.mp4")
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
            processed = preprocess_for_ocr(bus_roi)
            # processed = reduce_glow(bus_roi)
            cv2.imshow("wateva", processed)
            cv2.waitKey(0)

            text_results = reader.readtext(processed)

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