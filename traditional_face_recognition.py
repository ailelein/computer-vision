import cv2
import numpy as np

# Initialize the face detector (Haar cascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load and prepare the face recognizer (Eigenfaces)
recognizer = cv2.face.EigenFaceRecognizer_create()

# Prepare training data (images and labels)
def prepare_training_data():
    faces = []
    labels = []

    # Load images from the dataset
    for person_id in range(1, 3):  # Assuming you have 2 people
        for image_id in range(1, 4):  # 3 images per person
            image_path = f"dataset/person{person_id}_{image_id}.jpg"
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces_detected:
                face = gray[y:y + h, x:x + w]

                # Resize face to a fixed size (e.g., 200x200)
                face_resized = cv2.resize(face, (200, 200))  # Resize face to 200x200

                faces.append(face_resized)
                labels.append(person_id)

    return faces, labels

# Prepare the training data
faces, labels = prepare_training_data()

# Train the recognizer using Eigenfaces
recognizer.train(faces, np.array(labels))

# Save the trained model (optional)
recognizer.save("eigenface_trainer.yml")


# Now let's test the recognizer with a new image
def recognize_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the new image

    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces_detected:
        face = gray[y:y + h, x:x + w]

        # Resize the detected face to the same size as the training images (200x200)
        face_resized = cv2.resize(face, (200, 200))
        # Predict the identity of the detected face using the trained recognizer
        label, confidence = recognizer.predict(face_resized)

        # Draw the bounding box and label the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Person {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Predicted label: {label}, Confidence: {confidence}")
    # Display the result
    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



recognizer.read("eigenface_trainer.yml")  # Load the saved model
# Test recognition with a new image
recognize_face("12.jpg")
