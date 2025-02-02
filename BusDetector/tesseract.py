import cv2
import time  # Measure processing time
import pytesseract  # OCR for bus number recognition
from ultralytics import YOLO

# Tesseract Setup (Required on Windows - Update path if necessary)
# Uncomment and update the path if using Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file (or use `0` for webcam)
cap = cv2.VideoCapture("icHack/BusDetector/flipped_stock_footage/Bus_Footage_Cropped 20.mp4")

# Start timing
start_time = time.time()

frame_count = 0
bus_boxes = []
ocr_counter = 0
OCR_INTERVAL = 10  # Run OCR every 10 YOLO detections

# 🚀 Function: Preprocess image for better OCR performance
def preprocess_for_tesseract(image):
    """Convert image to grayscale and apply thresholding to improve OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply thresholding
    return binary

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
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

                if cls_id == 5 and conf > 0.5:  # Class ID 5 = Bus
                    bus_boxes.append((x1, y1, x2, y2))

        # 🚀 Run OCR every 10 YOLO detections
        if ocr_counter % OCR_INTERVAL == 0:
            for x1, y1, x2, y2 in bus_boxes:
                bus_roi = frame[y1:y2, x1:x2]  # Crop bus number area

                # Preprocess image for Tesseract OCR
                processed_roi = preprocess_for_tesseract(bus_roi)

                # Run Tesseract OCR with number filtering
                text = pytesseract.image_to_string(processed_roi, config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

                # Print detected text if it's not empty
                if text.strip():
                    print(f"🚍 Detected Bus Number: {text.strip()}")

        ocr_counter += 1

    # Show video with detection
    cv2.imshow("Bus Detection & Number Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Stop timer
end_time = time.time()
total_time = end_time - start_time  # Compute elapsed time

# Print total processing time
print(f"\n🚀 Total processing time: {total_time:.2f} seconds")

cap.release()
cv2.destroyAllWindows()
