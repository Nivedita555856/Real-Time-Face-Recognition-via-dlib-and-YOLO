import os
import cv2
import dlib
import numpy as np
from datetime import datetime
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cx_Oracle
import shutil  # Added for file moving
from ultralytics import YOLO
# ====== CONFIG ======
EMPLOYEE_IMAGES_PATH = "data/EmployeeImages/"
CAPTURE_FOLDER = "COMING_IN/"
CAPTURED_FACES_PATH = "recognized_faces/"
DELETED_FOLDER = "deleted/"  # Folder for unrecognized/repeated images
CSV_LOG_FILE = "attendance.csv"
RECOGNITION_THRESHOLD = 0.5
ATTENDANCE_COOLDOWN = 300  # 5 minutes between attendance records for same person

# Initialize models
detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

dsn = cx_Oracle.makedsn("172.15.2.242", 1521, sid="ORCL")
user = "TRAINING"
password = "training123"
cx_Oracle.init_oracle_client(
    lib_dir=r"C:\Users\rcf\Downloads\instantclient-basic-windows.x64-19.9.0.0.0dbru\instantclient_19_9")
try:
    conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
    print("Connection established")
except Exception as e:
    print(e)

# Create deleted folder if it doesn't exist
if not os.path.exists(DELETED_FOLDER):
    os.makedirs(DELETED_FOLDER)

# Load known faces
known_encodings = []
known_names = []

for file in os.listdir(EMPLOYEE_IMAGES_PATH):
    if file.endswith(('.jpg', '.png')):
        img = cv2.imread(os.path.join(EMPLOYEE_IMAGES_PATH, file))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(rgb, 1)
        if faces:
            shape = shape_predictor(rgb, faces[0])
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            known_encodings.append(encoding)
            known_names.append(os.path.splitext(file)[0])

# Attendance tracking
last_attendance = {}  # name: last_attendance_time

if not os.path.exists(CAPTURED_FACES_PATH):
    os.makedirs(CAPTURED_FACES_PATH)

class FaceHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.jpg'):
            return

        time.sleep(0.5)  # Wait for file write to complete
        self.process_face(event.src_path)

    def process_face(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                # self.move_to_deleted(image_path)
                print("No image detected")
                return

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(rgb, 1)

            if not faces:
                # self.move_to_deleted(image_path)
                print("No face detected")
                return

            shape = shape_predictor(rgb, faces[0])
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))

            matches = [np.linalg.norm(encoding - known) for known in known_encodings]
            min_distance = min(matches) if matches else None

            if min_distance is None or min_distance > RECOGNITION_THRESHOLD:
                # self.move_to_deleted(image_path)
                print("min distance is none or >reg_threshold")
                return

            best_index = matches.index(min_distance)
            name = known_names[best_index]

            # Extract timestamp from filename (assuming format like "capture_20230704_143022.jpg")
            try:
                filename = os.path.basename(image_path)
                # Extract the timestamp parts - adjust this based on your actual filename format
                # This assumes format: capture_YYYYMMDD_HHMMSS.jpg
                # time_str = filename.split('_')[1] + '_' + filename.split('_')[2].split('.')[0]
                time_str = filename.split('.')[0]
                print("error getting dt")
                capture_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                # Format for SQL insertion (Oracle typically uses TO_DATE with format string)
                sql_time = capture_time.strftime("%d-%b-%Y %H:%M:%S")
            except Exception as e:
                print(f"Could not extract time from filename: {str(e)}")
                capture_time = datetime.now()
                sql_time = capture_time.strftime("%d-%b-%Y %H:%M:%S")

            # Check attendance cooldown
            current_time = datetime.now()
            oracle_time_format = current_time.strftime("%d-%b-%Y %H:%M:%S")
            if name in last_attendance:
                time_since_last = (current_time - last_attendance[name]).total_seconds()
                if time_since_last < ATTENDANCE_COOLDOWN:
                    print(f"Skipping {name} - recently logged")
                    # self.move_to_deleted(image_path)
                    return

            last_attendance[name] = current_time

            # Save and log
            recognized_filename = f"{capture_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            recognized_path = os.path.join(CAPTURED_FACES_PATH, recognized_filename)
            cv2.imwrite(recognized_path, img)

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise ValueError("Could not encode image")

            img_bytes = buf.tobytes()  # Ready for BLOB column

            try:
                # Prepare and execute insert
                cur = conn.cursor()
                # insert_sql = """
                #              INSERT INTO CCTV_REG_OUT (EMPLOYEEID, OUTTIME, REG_IMAGE, CREATEDAT)
                #              VALUES (:1, TO_DATE(:2, 'DD-MON-YYYY HH24:MI:SS'), :3, \
                #                      TO_DATE(:4, 'DD-MON-YYYY HH24:MI:SS')) \
                #              """
                insert_sql = """
                             INSERT INTO CCTV_REG_IN (EMPLOYEEID, INTIME, REG_IMAGE, CREATEDAT)
                             VALUES (:1, TO_DATE(:2, 'DD-MON-YYYY HH24:MI:SS'), :3,
                                     TO_DATE(:4, 'DD-MON-YYYY HH24:MI:SS')) 
                             """
                # cur.execute(insert_sql, (name, sql_time, img_bytes, current_time))
                cur.execute(insert_sql, (name, sql_time, img_bytes,oracle_time_format))
                conn.commit()
                print("Row inserted and committed!")

            except cx_Oracle.DatabaseError as e:
                # Handle insert/commit problems
                error_obj, = e.args  # Oracle packs one tuple
                print("Insert/commit failed:")
                print("   ORA-", error_obj.code)
                print("   ", error_obj.message)
                conn.rollback()
                # self.move_to_deleted(image_path)
                return

            finally:
                # Always close the cursor
                if 'cur' in locals():
                    cur.close()

            print(f"Logged attendance for {name} at {capture_time}")

            # Remove processed file after successful recognition
            try:
                os.remove(image_path)
            except:
                pass

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # self.move_to_deleted(image_path)

    def move_to_deleted(self, image_path):
        """Move unrecognized or duplicate images to deleted folder"""
        try:
            filename = os.path.basename(image_path)
            dest_path = os.path.join(DELETED_FOLDER, filename)
            shutil.move(image_path, dest_path)
            print(f"Moved unrecognized/duplicate image to {dest_path}")
        except Exception as e:
            print(f"Could not move file to deleted folder: {str(e)}")
            try:
                os.remove(image_path)
            except:
                pass


if __name__ == "__main__":
    print("Starting face recognition service...")

    # Process any existing files
    for file in os.listdir(CAPTURE_FOLDER):
        if file.endswith('.jpg'):
            FaceHandler().process_face(os.path.join(CAPTURE_FOLDER, file))

    # Set up file watcher
    event_handler = FaceHandler()
    observer = Observer()
    observer.schedule(event_handler, path=CAPTURE_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()