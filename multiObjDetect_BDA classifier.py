import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle
import random

# Paths to datasets
data_dir = "C:/Users/gbind/PycharmProjects/learning/dataset/"
categories = ["E-Waste", "Non-E-Waste"]

# Feature extraction for both file paths and NumPy arrays
def extract_features(image_or_path):
    # Check if input is a file path or an image array
    if isinstance(image_or_path, str):  # If it's a file path
        image = cv2.imread(image_or_path)
    else:  # If it's a NumPy array (e.g., sliding window input)
        image = image_or_path

    if image is not None:
        image = cv2.resize(image, (64, 64))  # Resize to 64x64
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()  # Flatten to 1D
    return None

# Parallelized feature extraction
def load_dataset(data_dir, categories):
    filepaths = []
    labels = []
    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        for filename in os.listdir(folder):
            filepaths.append(os.path.join(folder, filename))
            labels.append(label)

    # Parallel processing of features
    features = [extract_features(fp) for fp in filepaths]
    features, labels = zip(*[(f, l) for f, l in zip(features, labels) if f is not None])
    return np.array(features), np.array(labels)

print("Loading dataset...")
features, labels = load_dataset(data_dir, categories)

# Dimensionality Reduction with PCA
print("Applying PCA...")
pca = PCA(n_components=100)  # Reduce to 100 dimensions
features = pca.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate additional metrics
print("Generating classification report...")
report = classification_report(y_test, y_pred, target_names=categories)
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the Accuracy vs. Training Size
train_sizes = np.linspace(0.1, 0.9999999)  # Now it's a fraction (not an absolute number)
train_accuracies = []

for size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    clf.fit(X_train_subset, y_train_subset)
    y_train_pred = clf.predict(X_test)
    train_accuracies.append(accuracy_score(y_test, y_train_pred))

# Plotting Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plotting Accuracy vs Training Size
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_accuracies, marker='o', color='b', linestyle='-', linewidth=2)
plt.title("Accuracy vs. Training Size")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Plotting Precision, Recall, and F1-Score
metrics = classification_report(y_test, y_pred, output_dict=True)

# Corrected to access metrics using integer labels
precision = [metrics[str(label)]["precision"] for label in range(len(categories))]
recall = [metrics[str(label)]["recall"] for label in range(len(categories))]
f1_score = [metrics[str(label)]["f1-score"] for label in range(len(categories))]

plt.figure(figsize=(8, 6))
x = np.arange(len(categories))
width = 0.2

plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1_score, width, label="F1-Score")

plt.xlabel("Classes")
plt.ylabel("Scores")
plt.title("Precision, Recall, and F1-Score")
plt.xticks(x, categories)
plt.legend()
plt.show()

# Save the trained model and PCA object
with open("e_waste_classifier.pkl", "wb") as f:
    pickle.dump({"model": clf, "pca": pca, "categories": categories}, f)
print("Model saved as e_waste_classifier.pkl")

# Real-Time Object Detection with Max 3 Bounding Boxes
print("Starting real-time object detection...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    step_size = 50
    window_size = (100, 100)

    # Track the number of boxes drawn
    boxes_drawn = 0
    detections = []  # List to track detected objects and their confidence

    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours (which will correspond to objects in the image)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Only consider contours that are large enough to be considered objects
        if cv2.contourArea(contour) < 500:  # Minimum area for detection
            continue

        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        window = frame[y:y + h, x:x + w]

        # Extract features from the window
        features = extract_features(window)
        if features is not None:
            features = pca.transform([features])  # Apply PCA
            prediction = clf.predict(features)
            confidence = np.max(clf.predict_proba(features))  # Get the confidence (probability)
            label = categories[prediction[0]]

            # Only draw box if the confidence is above a certain threshold
            if confidence > 0.8:  # Adjust confidence threshold if necessary
                detections.append((x, y, label, confidence))

                # Draw bounding box and label with confidence
                color = (0, 255, 0) if label == "E-Waste" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                boxes_drawn += 1  # Increment count of boxes drawn

        if boxes_drawn >= 3:
            break  # Exit the loop if 3 boxes are already drawn

    # Display the result
    cv2.imshow("Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
