import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
import json

# Define file paths
root = 'C:/Users/gbind/PycharmProjects/learning/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification-main/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification-main/dataset/'

# Create a list to hold image paths and their corresponding class labels
data = []
for i in os.listdir(root):
    class_dir = os.path.join(root, i)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            data.append([image_path, i])

# Create DataFrame from the list
data = pd.DataFrame(data, columns=['path', 'class_'])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Split data into train, validation, and test sets
def split_data(data, ratio):
    last = int(len(data) * ratio)
    return data[:last], data[last:]

train, test = split_data(data, .8)
validation, train = split_data(train, .08)

# Save CSV files
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
validation.to_csv('valid.csv', index=False)

# Image preprocessing
size = 384
batch_size = 4
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=[0.8, 1.2],  # Added brightness variation
    channel_shift_range=50.0  # Added channel shift
)

train_images = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='path',
    y_col='class_',
    batch_size=batch_size,
    target_size=(size, size),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=42,
)

valid_generator = ImageDataGenerator(rescale=1.0 / 255)
valid_images = valid_generator.flow_from_dataframe(
    dataframe=validation,
    x_col='path',
    y_col='class_',
    target_size=(size, size),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
)

test_generator = ImageDataGenerator(rescale=1.0 / 255)
test_images = test_generator.flow_from_dataframe(
    dataframe=test,
    x_col='path',
    y_col='class_',
    target_size=(size, size),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
)

# Load EfficientNetB0 model as a feature extractor
def build_model(input_shape=(size, size, 3), num_classes=len(train_images.class_indices)):
    inputs = Input(shape=input_shape)

    # Use EfficientNetB0 pre-trained weights for feature extraction
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Unfreeze the base model layers

    # Fine-tune from a specific layer onward
    fine_tune_at = 100  # Fine-tune from the 100th layer onward (or adjust this as needed)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Dropout layer to avoid overfitting
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, output)

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])

    return model

# Build and train the model
model = build_model()

# Set up early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_images,
    validation_data=valid_images,
    epochs=20,
    steps_per_epoch=train_images.n // batch_size,
    validation_steps=valid_images.n // batch_size,
    callbacks=[early_stopping],
)

# Save the model
model.save('model.h5')

# Save the class indices to a JSON file
with open('class_indices.json', 'w') as f:
    json.dump(train_images.class_indices, f)

# Plot accuracy and loss curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train-accuracy', 'validation-accuracy'], loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train-loss', 'validation-loss'], loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images)
print(f"Test Accuracy: {test_accuracy}")
