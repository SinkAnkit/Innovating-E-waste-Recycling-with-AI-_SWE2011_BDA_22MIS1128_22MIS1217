import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.ops import box_iou

# File paths
root_dir = '/content/dataset/'
annotation_file = '/content/annotations.csv'

# Read annotations
print("Loading annotations...")
annotations = pd.read_csv(annotation_file)
print(f"Annotations loaded. Total entries: {len(annotations)}")

# Custom Dataset for Object Detection
class EWasteDataset(Dataset):
    def __init__(self, annotations, root_dir, transforms=None):
        self.annotations = annotations
        self.root_dir = root_dir
        self.transforms = transforms
        self.label_map = {label: i for i, label in enumerate(annotations['label'].unique())}
        print(f"Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.annotations['image_path'].unique())

    def __getitem__(self, idx):
        image_path = self.annotations['image_path'].unique()[idx]
        full_path = os.path.join(self.root_dir, image_path)
        image = Image.open(full_path).convert("RGB")

        boxes = []
        labels = []
        subset = self.annotations[self.annotations['image_path'] == image_path]
        for _, row in subset.iterrows():
            boxes.append([row['x_min'], row['y_min'], row['x_max'], row['y_max']])
            labels.append(self.label_map[row['label']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target

# Load dataset
print("Creating dataset...")
dataset = EWasteDataset(annotations=annotations, root_dir=root_dir, transforms=F.to_tensor)
print(f"Dataset created. Total images: {len(dataset)}")

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(f"Dataset split into {train_size} training samples and {test_size} test samples.")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
print("DataLoaders created.")

# Load the Faster R-CNN model
print("Loading Faster R-CNN model...")
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(annotations['label'].unique()) + 1  # +1 for background
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features,
    num_classes
)
print(f"Model loaded with {num_classes} classes.")

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
print(f"Training setup complete. Using device: {device}")

# Training loop
num_epochs = 1
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss:.4f}")

# Evaluation with metrics
def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    total_true_positives = 0
    total_predictions = 0
    total_ground_truths = 0
    all_labels = []
    all_preds = []

    print("Evaluating model...")
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            predictions = model(images)

        for i, prediction in enumerate(predictions):
            pred_boxes = prediction["boxes"].cpu()
            pred_labels = prediction["labels"].cpu()
            scores = prediction["scores"].cpu()  # Confidence scores
            gt_boxes = targets[i]["boxes"].cpu()
            gt_labels = targets[i]["labels"].cpu()

            if len(pred_boxes) == 0:
                continue  # Skip if no predictions are made

            # Filter predictions based on confidence score
            valid_indices = scores > 0.3
            pred_boxes = pred_boxes[valid_indices]
            pred_labels = pred_labels[valid_indices]
            scores = scores[valid_indices]

            if len(pred_boxes) == 0:
                continue  # If no valid predictions after filtering

            total_predictions += len(pred_boxes)
            total_ground_truths += len(gt_boxes)

            # To avoid overcounting, track which ground truth boxes are matched
            matched_ground_truths = []

            if len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)

                for j in range(len(pred_boxes)):
                    matched = False
                    # Try to match each predicted box to a ground truth box
                    for k in range(len(gt_boxes)):
                        if ious[j, k] > iou_threshold and k not in matched_ground_truths:
                            matched_ground_truths.append(k)
                            matched = True
                            break

                    if matched:
                        total_true_positives += 1

            # Collect labels for confusion matrix
            all_labels.extend(gt_labels.numpy())
            all_preds.extend(pred_labels.numpy())

    # Avoid division by zero
    precision = total_true_positives / total_predictions if total_predictions > 0 else 0
    recall = total_true_positives / total_ground_truths if total_ground_truths > 0 else 0
    accuracy = total_true_positives / total_ground_truths if total_ground_truths > 0 else 0

    mean_precision = precision
    mean_recall = recall

    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    if len(all_labels) > 0 and len(all_preds) > 0:
        min_len = min(len(all_labels), len(all_preds))
        cm = confusion_matrix(all_labels[:min_len], all_preds[:min_len])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    return mean_precision, mean_recall, accuracy

# Run evaluation
evaluate_model(model, test_loader, device)
