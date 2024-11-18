import cv2
import face_recognition
import pandas as pd
from datetime import datetime

# Initialize the attendance DataFrame
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# Load known face images and names
known_face_encodings = []
known_face_names = []

# Add images of individuals to the database
known_images = {
    "Alice": "alice.jpg",  # Replace with the path to Alice's photo
    "Bob": "bob.jpg",      # Replace with the path to Bob's photo
}

for name, image_path in known_images.items():
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Track attendance
attendance_log = {name: False for name in known_face_names}

def mark_attendance(name):
    """Mark attendance in the DataFrame."""
    if not attendance_log[name]:  # Mark attendance only once
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        attendance_df.loc[len(attendance_df)] = [name, timestamp]
        attendance_log[name] = True
        print(f"{name} marked present at {timestamp}")

# Start the webcam feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the closest match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Mark attendance
        if name != "Unknown":
            mark_attendance(name)

        # Draw a rectangle around the face and label it
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back to original size
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Attendance', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Save attendance log to a CSV file
attendance_df.to_csv("attendance_log.csv", index=False)
print("Attendance log saved to attendance_log.csv")




import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the attendance DataFrame
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# List of registered names (Simulated for simplicity)
registered_names = ["Alice", "Bob", "Charlie"]
attendance_log = {name: False for name in registered_names}  # Track attendance

def mark_attendance(name):
    """Mark attendance in the DataFrame."""
    if not attendance_log[name]:  # Mark attendance only once
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

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box for the face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw bounding box and keypoints
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mp_drawing.draw_detection(frame, detection)

            # Simulate name assignment (replace this with real face recognition if needed)
            name = registered_names[0]  # For demonstration, assigning the first name

            # Display name on the frame
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mark attendance for the detected person
            mark_attendance(name)

    # Display the resulting frame
    cv2.imshow('Real-Time Attendance', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Save attendance log to a CSV file
attendance_df.to_csv("attendance_log_mediapipe.csv", index=False)
print("Attendance log saved to attendance_log_mediapipe.csv")
