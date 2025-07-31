# 🎵 Music Genre Classifier + Recommender

This project is a full-stack machine learning and deep learning application that classifies the **genre** of an uploaded `.wav` file and recommends **similar songs** based on audio features. It also fetches **top YouTube search results** based on the predicted genre.

---

## 🚀 Features

- 🎧 Upload `.wav` audio files
- 🎼 Predict music genre using a CNN model
- 🔍 Recommend similar songs using content-based filtering
- 📺 Fetch YouTube results for discovered genres using the YouTube API
- 🧠 End-to-end audio pipeline (Spectrogram → CNN → Prediction → Recommendation)

---

## 🗂️ Project Structure

```bash
music_genre_cnn/
│
├── content_based/
│   ├── extract_features.py
│   ├── generate_feature_dataset.py
│   ├── recommend_similar.py
│   └── features_dataset.csv
│
├── embeddings/
│   ├── filenames.npy
│   ├── label_encoder.pkl
│   └── embeddings.npy
│
├── genres/                  # Folders with genre-wise .wav files (blues, classical, etc.)
│
├── models/
│   └── music_genre_cnn.h5
│
├── spectrograms/
│   └── [generated .png or .npy files for CNN input]
│
├── cnn_predictor.py
├── create_label_map.py
├── extract_embeddings.py
├── generate_spectrograms.py
├── create_label_map.json
├── predict_genre.py
├── recommend_similar.py
├── streamlit_app.py         # 🌐 Main front-end app
├── train_model.py
├── yt_search.py             # 🔎 YouTube API integration
│
├── requirements.txt
├── .gitignore
├── README.md               
└── LICENSE                  
