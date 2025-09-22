import os
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO

# ====== CONFIG ======
CAPTURE_FOLDER = "COMING_IN/"

RTSP_URL = ''

MIN_FACE_CONFIDENCE = 0.6  # Minimum confidence for face detection
SHOW_OUTPUT = True
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
MIN_FACE_SIZE = 0  # Minimum face width/height in pixels
MAX_FACE_SIZE = 1000  # Maximum face width/height
BLUR_THRESHOLD = 40.0  # Minimum sharpness threshold

# Initialize YOLOv8 model
yolo_model = YOLO('yolov8n.pt')
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


def initialize_stream():
    """Initialize video stream using the working method from Code 1"""
    max_attempts = 3
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture()

        # Try multiple URL variations (from Code 1)
        url_variants = [
            RTSP_URL,  # Original URL
            RTSP_URL.replace('?channel=1&subtype=0', ''),  # Without params
            RTSP_URL.replace('cam/realmonitor', 'Streaming/Channels/101'),  # Alternative
            RTSP_URL.split('@')[0] + '@172.22.9.119:554/Streaming/Channels/101'  # Simplified
        ]

        for url in url_variants:
            print(f"Attempt {attempt + 1}: Trying URL: {url}")

            # Set important properties BEFORE opening (from Code 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_FOURCC, 0x48323634)  # 'h264'

            if cap.open(url, cv2.CAP_FFMPEG):
                # Verify we can actually read a frame
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully opened: {url}")
                    return cap
                else:
                    print("Opened but couldn't read frame")
                    cap.release()

            print(f"Failed with error: {cap.get(cv2.CAP_PROP_BACKEND)}")
            time.sleep(2)

    print("All connection attempts failed")
    return None


def is_clear_face(face_roi):
    """Check if face meets quality standards (from Code 1)"""
    try:
        if face_roi.size == 0:
            return False

        # Size check
        height, width = face_roi.shape[:2]
        if (width < MIN_FACE_SIZE or height < MIN_FACE_SIZE or
                width > MAX_FACE_SIZE or height > MAX_FACE_SIZE):
            return False

        # Blur check
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() > BLUR_THRESHOLD
    except:
        return False


def detect_faces_yolo(frame):
    """Precise face detection with optimized head cropping"""
    try:
        # Load specialized face model (lazy initialization)
        if not hasattr(detect_faces_yolo, 'face_model'):
            detect_faces_yolo.face_model = YOLO('yolov8n-face.pt')  # More accurate than person detection

        # Detect faces with confidence threshold
        results = detect_faces_yolo.face_model(frame, verbose=False, conf=max(0.3, MIN_FACE_CONFIDENCE))

        faces = []
        for result in results:
            for box in result.boxes:
                # Get raw coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Calculate dynamic padding based on face size
                face_width = x2 - x1
                padding = int(face_width * 0.15)  # Smarter padding (15% of face width)

                # Apply intelligent cropping
                crop_top = int(padding * 0.7)  # Less padding top (forehead)
                crop_bottom = int(padding * 1.3)  # More padding bottom (chin)

                # Final face coordinates
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - crop_top)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + crop_bottom)

                # Calculate aspect ratio and enforce square-ish crop
                current_ratio = (x2 - x1) / (y2 - y1)
                if current_ratio > 1.2:  # Too wide
                    center_y = (y1 + y2) // 2
                    desired_height = int((x2 - x1) / 1.1)
                    y1 = center_y - desired_height // 2
                    y2 = center_y + desired_height // 2
                elif current_ratio < 0.8:  # Too tall
                    center_x = (x1 + x2) // 2
                    desired_width = int((y2 - y1) * 0.9)
                    x1 = center_x - desired_width // 2
                    x2 = center_x + desired_width // 2

                faces.append((x1, y1, x2, y2))

        return faces

    except Exception as e:
        print(f"Face detection error: {e}")
        return []


def save_face(face_img):
    """Save face image with timestamp"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{CAPTURE_FOLDER}{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        return True
    except Exception as e:
        print(f"Error saving face: {e}")
        return False


def main():
    cap = initialize_stream()
    if not cap:
        return

    frame_count = 0
    last_valid_frame = None

    while True:
        start_time = time.time()

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Frame read error, reconnecting...")
            time.sleep(1)
            cap.release()
            cap = initialize_stream()
            if not cap:
                break
            continue

        frame_count += 1
        last_valid_frame = frame.copy()

        # Skip processing if not our target frame
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            if SHOW_OUTPUT:
                cv2.imshow("Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # Detect faces using YOLOv8
        faces = detect_faces_yolo(frame)

        # Process detected faces
        for (x1, y1, x2, y2) in faces:
            face_roi = frame[y1:y2, x1:x2]
            if is_clear_face(face_roi):
                if save_face(face_roi):
                    print(f"Saved face at {datetime.now().strftime('%H:%M:%S')}")

        # Display processing info
        if SHOW_OUTPUT:
            display_frame = frame.copy()
            for (x1, y1, x2, y2) in faces:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Faces: {len(faces)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if SHOW_OUTPUT:
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
