import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, learning_curve
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to datasets
data_dir = "C:/Users/gbind/PycharmProjects/learning/dataset/"
categories = ["E-Waste", "Non-E-Waste"]

# Feature extraction for both file paths and NumPy arrays
def extract_features(image_or_path):
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

# Train SVM Classifier
print("Training SVM Classifier...")
clf = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

metrics = ['Precision', 'Recall', 'F1-Score']
e_waste_metrics = [precision[0], recall[0], f1[0]]
non_e_waste_metrics = [precision[1], recall[1], f1[1]]

x = np.arange(len(metrics))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, e_waste_metrics, width, label='E-Waste')
ax.bar(x + width/2, non_e_waste_metrics, width, label='Non-E-Waste')

ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, F1-Score for E-Waste and Non-E-Waste')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Learning Curves
train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy", color='blue')
plt.plot(train_sizes, test_mean, label="Test Accuracy", color='green')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve (Accuracy vs Training Set Size)')
plt.legend()
plt.show()

# Save the trained model and PCA object
with open("e_waste_classifier_svm.pkl", "wb") as f:
    pickle.dump({"model": clf, "pca": pca, "categories": categories}, f)
print("Model saved as e_waste_classifier_svm.pkl")

# Real-Time Object Detection with Max 3 Bounding Boxes
print("Starting real-time object detection...")
cap = cv2.VideoCapture(0)
detections = []  # List to track detected objects and their confidence

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
        if cv2.contourArea(contour) < 200:  # Lowering the threshold for contour area
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

# Prediction Confidence Histogram
confidences = [d[3] for d in detections]
plt.figure(figsize=(8, 6))
plt.hist(confidences, bins=20, color='orange', edgecolor='black')
plt.title("Distribution of Prediction Confidence")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()
