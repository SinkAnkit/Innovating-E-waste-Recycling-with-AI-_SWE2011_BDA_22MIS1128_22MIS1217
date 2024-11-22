import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json

# Load the trained model and class indices
model = tf.keras.models.load_model('model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Resize frame to match the input size of the model
    img = cv2.resize(frame, (384, 384))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    # Predict the class
    predictions = model.predict(img)

    # Get the top predicted class and its confidence
    predicted_class = np.argmax(predictions[0])  # Index of the highest confidence class
    predicted_label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class]

    # Display the predicted class and confidence (first line)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'{predicted_label}: {confidence:.2f}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
