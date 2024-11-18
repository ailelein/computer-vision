import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the attendance DataFrame
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# Load known face images and extract features (using simple histograms for demo purposes)
def extract_features(image_path):
    """Extract features from an image for face recognition."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

# Database of known individuals
known_images = {
    "Alice": "alice.jpg",  # Replace with actual paths
    "Bob": "bob.jpg",      # Replace with actual paths
}
known_face_features = {name: extract_features(path) for name, path in known_images.items()}

attendance_log = {name: False for name in known_images.keys()}

def mark_attendance(name):
    """Mark attendance in the DataFrame."""
    if not attendance_log[name]:  # Mark attendance only once
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        attendance_df.loc[len(attendance_df)] = [name, timestamp]
        attendance_log[name] = True
        print(f"{name} marked present at {timestamp}")

def recognize_face(detected_face):
    """Match a detected face against known faces."""
    # Convert the detected face to grayscale
    gray_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
    face_features = cv2.calcHist([gray_face], [0], None, [256], [0, 256]).flatten()

    # Compare histograms (you can replace this with more advanced methods)
    min_distance = float("inf")
    recognized_name = "Unknown"
    for name, features in known_face_features.items():
        distance = np.linalg.norm(face_features - features)
        if distance < min_distance and distance < 1e7:  # Threshold to avoid false positives
            min_distance = distance
            recognized_name = name

    return recognized_name

# Start webcam feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Crop the face region
            face_region = frame[y:y + h, x:x + w]

            # Recognize face
            if face_region.size > 0:
                name = recognize_face(face_region)
                mark_attendance(name)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Attendance", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Save attendance log to a CSV file
attendance_df.to_csv("attendance_log_with_uploaded_images.csv", index=False)
print("Attendance log saved to attendance_log_with_uploaded_images.csv")






import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize attendance DataFrame
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# Known faces data (simulating recognition with pre-determined names for detected faces)
# In a real implementation, you can compare face landmarks or use another library for recognition.
known_faces = {
    "Alice": {"box": (0.3, 0.3, 0.7, 0.7)},  # Example: Replace with a matching algorithm
    "Bob": {"box": (0.5, 0.5, 0.9, 0.9)},
}

attendance_log = {name: False for name in known_faces}

def mark_attendance(name):
    """Mark attendance in the DataFrame."""
    if not attendance_log[name]:
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        attendance_df.loc[len(attendance_df)] = [name, timestamp]
        attendance_log[name] = True
        print(f"{name} marked present at {timestamp}")

# Start webcam feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Here we simply assume the first detection is "Alice" for demo purposes
            # Replace this with your logic for actual face comparison
            name = "Alice"

            mark_attendance(name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Attendance", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
attendance_df.to_csv("attendance_log_mediapipe.csv", index=False)
print("Attendance log saved to attendance_log_mediapipe.csv")


