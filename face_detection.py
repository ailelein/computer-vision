import cv2
import pandas as pd
from datetime import datetime

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a DataFrame to store attendance
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# List of people for attendance (Add more as needed)
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
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Simulate name recognition by assigning a name from the list (for simplicity)
        # In a real project, use face recognition techniques.
        name = registered_names[0]  # Default to the first name for demonstration

        # Display name on the frame
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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
attendance_df.to_csv("attendance_log.csv", index=False)
print("Attendance log saved to attendance_log.csv")
