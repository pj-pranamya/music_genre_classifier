import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths
AUDIO_DIR = 'genres'
SPEC_DIR = 'spectrograms'

# Ensure output directory exists
os.makedirs(SPEC_DIR, exist_ok=True)

# Function to create and save spectrogram
def save_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path)  # Removed duration=30 to load full audio
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Loop through each genre folder
for genre in os.listdir(AUDIO_DIR):
    genre_path = os.path.join(AUDIO_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    output_genre_path = os.path.join(SPEC_DIR, genre)
    os.makedirs(output_genre_path, exist_ok=True)

    for filename in os.listdir(genre_path):
        if filename.endswith('.wav'):
            audio_file = os.path.join(genre_path, filename)
            output_file = os.path.join(output_genre_path, filename.replace('.wav', '.png'))

            try:
                print(f"🎵 Processing {audio_file}...")
                save_mel_spectrogram(audio_file, output_file)
            except Exception as e:
                print(f"❌ Failed to process {audio_file}: {e}")
