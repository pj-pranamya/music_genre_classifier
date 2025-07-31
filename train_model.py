import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
DATASET_DIR = 'spectrograms'
IMG_SIZE = 128

# Load data
data = []
labels = []

for genre in os.listdir(DATASET_DIR):
    genre_path = os.path.join(DATASET_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    for img_file in os.listdir(genre_path):
        if img_file.endswith('.png'):
            img_path = os.path.join(genre_path, img_file)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(genre)

# Shuffle and preprocess
data = np.array(data) / 255.0
labels = np.array(labels)

le = LabelEncoder()
labels_encoded = to_categorical(le.fit_transform(labels))

X_train, X_val, y_train, y_val = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Build Functional CNN model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu', name="embedding_layer")(x)
x = Dropout(0.3)(x)
outputs = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Save model and label encoder
os.makedirs("models", exist_ok=True)
model.save('models/music_genre_cnn.h5')
np.save('models/label_classes.npy', le.classes_)

print("✅ Model training complete and saved.")
