import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from content_based.extract_features import extract_features

def recommend_similar_songs(file_path, dataset_csv='content_based/features_dataset.csv', top_n=5):
    # Load the dataset
    df = pd.read_csv(dataset_csv)
    if df.empty:
        print("❌ Feature dataset is empty!")
        return []

    feature_columns = df.columns[2:]  # Exclude filename and genre
    song_features = extract_features(file_path)
    
    if song_features is None:
        print("❌ Could not extract features from input song.")
        return []

    if len(song_features) != len(feature_columns):
        print("❌ Feature length mismatch.")
        return []

    input_df = pd.DataFrame([song_features], columns=feature_columns)

    # Calculate cosine similarity
    similarities = cosine_similarity(input_df, df[feature_columns])[0]
    df['similarity'] = similarities

    # Sort and return top matches
    recommended = df.sort_values(by='similarity', ascending=False)
    return [(row['filename'], row['similarity']) for _, row in recommended.head(top_n).iterrows()]
