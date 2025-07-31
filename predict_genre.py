# content_based/extract_features.py

import os
import numpy as np
import librosa

def extract_features(file_path, sr=22050, duration=30):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=sr, duration=duration)

        # Compute MFCCs (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Compute Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel, axis=1)

        # Concatenate all features into one vector
        features = np.concatenate((mfcc_mean, chroma_mean, mel_mean))

        return features

    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None
