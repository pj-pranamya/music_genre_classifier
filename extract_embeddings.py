import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
SPECTROGRAM_FOLDER = 'spectrograms'
MODEL_PATH = 'models/music_genre_cnn.h5'
EMBEDDINGS_PATH = 'embeddings/embeddings.npy'
LABELS_PATH = 'embeddings/filenames.npy'
ENCODER_PATH = 'embeddings/label_encoder.pkl'

# Load trained model
print("🔄 Loading trained model...")
model = load_model(MODEL_PATH)

# Build the model (needed before slicing layers)
dummy_input = np.random.rand(1, 128, 128, 3)
model.predict(dummy_input)

# Create embedding model (remove final classification layer)
print("🧠 Creating embedding model...")
embedding_model = Sequential(model.layers[:-1])

# Extract embeddings
print("📥 Extracting spectrograms and labels...")
X = []
filenames = []

for genre in os.listdir(SPECTROGRAM_FOLDER):
    genre_path = os.path.join(SPECTROGRAM_FOLDER, genre)
    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if file.endswith('.png'):
            img_path = os.path.join(genre_path, file)
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            X.append(img_array)
            filenames.append(f"{genre}/{file}")

X = np.array(X)

# Encode labels (optional but useful later)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform([f.split("/")[0] for f in filenames])
os.makedirs('embeddings', exist_ok=True)
joblib.dump(label_encoder, ENCODER_PATH)

# Extract embeddings
print("🧬 Generating embeddings...")
embeddings = embedding_model.predict(X, batch_size=32, verbose=1)

# Save embeddings and corresponding filenames
np.save(EMBEDDINGS_PATH, embeddings)
np.save(LABELS_PATH, np.array(filenames))

print("✅ Embeddings and filenames saved to 'embeddings/' folder.")
