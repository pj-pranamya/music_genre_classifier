# cnn_predictor.py

import librosa
import librosa.display
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 128
MODEL_PATH = 'models/music_genre_cnn.h5'
LABELS_PATH = 'models/label_classes.npy'

# Load model and labels once
model = load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH)

def extract_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    img = cv2.imread('temp.png')
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

def predict_genre(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ File '{audio_path}' not found.")

    spectrogram = extract_spectrogram(audio_path)
    prediction = model.predict(spectrogram)
    predicted_genre = label_classes[np.argmax(prediction)]
    return predicted_genre
