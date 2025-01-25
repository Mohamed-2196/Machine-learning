import pandas as pd
import numpy as np
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models

##get the images folder or copy the code and put it in kaggle
train_df = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/train.csv')
image_folder = '/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/Images/Images/'

print("Columns in DataFrame:", train_df.columns.tolist())

train_df.columns = train_df.columns.str.strip()

images = []
labels = []

for index, row in train_df.iterrows():
    try:
        img_path = os.path.join(image_folder, row['name'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Image at {img_path} could not be read.")
            continue
        image = cv2.resize(image, (128, 128))
        images.append(image)
        labels.append(row[1:].values.astype(int))
    except KeyError:
        print(f"KeyError: 'name' not found in row {index}. Available columns: {train_df.columns.tolist()}")
        continue

images = np.array(images) / 255.0
labels = np.array(labels, dtype=int)

print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels[0]), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(images, labels, epochs=10, batch_size=32)

test_df = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/test.csv')
test_images = []

for index, row in test_df.iterrows():
    try:
        img_path = os.path.join(image_folder, row['name'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Image at {img_path} could not be read.")
            continue
        image = cv2.resize(image, (128, 128))
        test_images.append(image)
    except KeyError:
        print(f"KeyError: 'name' not found in row {index}. Available columns: {test_df.columns.tolist()}")
        continue

test_images = np.array(test_images) / 255.0

predictions = model.predict(test_images)
predictions_binary = (predictions > 0.5).astype(int)

output_df = pd.DataFrame(predictions_binary, columns=train_df.columns[1:])
output_df.insert(0, 'name', test_df['name'])
output_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")