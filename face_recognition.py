import cv2
import tensorflow as tf
import numpy as np

# --------------------------------
# Load trained CNN model
# --------------------------------

model = tf.keras.models.load_model("face_model.keras")

# --------------------------------
# Load class names
# --------------------------------

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Classes:", class_names)

# --------------------------------
# Load Haar Cascade
# --------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------
# Start webcam
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    # Capture frame
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process each detected face
    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Resize face
        face_resized = cv2.resize(
            face_rgb,
            (128, 128)
        )

        # Convert to array
        face_array = np.array(face_resized)

        # Add batch dimension
        face_array = np.expand_dims(face_array, axis=0)

        # CNN prediction
        prediction = model.predict(
            face_array,
            verbose=0
        )

        predicted_index = np.argmax(prediction[0])

        predicted_name = class_names[predicted_index]

        confidence = prediction[0][predicted_index] * 100

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Display prediction
        text = "{} {:.2f}%".format(
            predicted_name,
            confidence
        )

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display webcam
    cv2.imshow(
        "Face Recognition using CNN",
        frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------
# Release resources
# --------------------------------

camera.release()
cv2.destroyAllWindows()