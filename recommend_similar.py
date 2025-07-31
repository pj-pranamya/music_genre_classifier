import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from tensorflow.keras.models import load_model, Model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Load model and build embedding extractor
print("🔄 Loading CNN model...")
model = load_model('models/music_genre_cnn.h5')
embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Load saved embeddings and corresponding filenames
print("📂 Loading embeddings and labels...")
embeddings = np.load('embeddings/embeddings.npy')
filenames = np.load('embeddings/filenames.npy', allow_pickle=True)

# Convert audio to embedding
def audio_to_embedding(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    temp_path = 'temp_query.png'
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = cv2.imread(temp_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return embedding_model.predict(img)

# Recommend top-N similar tracks
def recommend(audio_path, top_n=5):
    print(f"\n🎵 Processing query audio: {audio_path}")
    query_embedding = audio_to_embedding(audio_path)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    print("\n🎧 Recommended Similar Songs:")
    for i in top_indices:
        print(f"✔ {filenames[i]} (Similarity: {similarities[i]:.2f})")

# Example usage
recommend('test_music.wav')  # 🔁 Replace with actual .wav file path
