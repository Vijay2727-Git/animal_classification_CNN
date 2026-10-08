import tensorflow as tf
import numpy as np
import os
import csv

# Load trained model
model = tf.keras.models.load_model("face_model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Path to test images folder
test_folder = r"C:\Users\vragu\Documents\project\Face_Recognition_CNN\test_images"

# Save predictions to CSV
with open("results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "predicted_class", "confidence", "top3_predictions"])

    for file_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, file_name)

        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".png")):
            continue

        image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
        image_array = tf.keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        probs = prediction[0]

        # Top prediction
        predicted_index = np.argmax(probs)
        predicted_name = class_names[predicted_index]
        confidence = probs[predicted_index] * 100

        # Top 3 predictions
        top_indices = probs.argsort()[-3:][::-1]
        top3 = [f"{class_names[i]}: {probs[i]*100:.2f}%" for i in top_indices]

        print(f"{file_name} → {predicted_name} ({confidence:.2f}%)")
        print("Top 3:", ", ".join(top3), "\n")

        writer.writerow([file_name, predicted_name, f"{confidence:.2f}%", "; ".join(top3)])
