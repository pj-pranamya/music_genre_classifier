import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from content_based.extract_features import extract_features  # ✅ Import your improved function

# Settings
DATASET_DIR = "genres"
CSV_OUTPUT_PATH = "content_based/features_dataset.csv"

def build_feature_dataset():
    all_data = []

    for genre in os.listdir(DATASET_DIR):
        genre_path = os.path.join(DATASET_DIR, genre)
        if not os.path.isdir(genre_path):
            continue

        for filename in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            if not filename.endswith(".wav"):
                continue

            file_path = os.path.join(genre_path, filename)
            features = extract_features(file_path)

            if features is not None:
                data = {
                    "filename": filename,
                    "genre": genre,
                    **{f"feature_{i+1}": features[i] for i in range(len(features))}
                }
                all_data.append(data)

    df = pd.DataFrame(all_data)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"✅ Feature dataset saved to {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    build_feature_dataset()
