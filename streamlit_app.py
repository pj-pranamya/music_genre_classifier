import streamlit as st
import os
from cnn_predictor import predict_genre  # Your CNN prediction module
from content_based.recommend_similar import recommend_similar_songs
from content_based.extract_features import extract_features
from youtubesearchpython import VideosSearch

st.set_page_config(page_title="Music Genre & Recommender", layout="centered")
st.title("🎵 Music Genre & Mood Classifier + Recommender")
st.write("Upload a `.wav` file to predict its genre and get recommendations.")

uploaded_file = st.file_uploader("🎼 Upload a music file (.wav)", type=["wav"])

if uploaded_file is not None:
    file_path = "uploaded_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path, format="audio/wav")
    st.success("✅ File uploaded successfully!")

    if st.button("🎧 Predict Genre & Recommend"):
        try:
            # === Genre Prediction ===
            predicted_genre = predict_genre(file_path)
            st.subheader("🎯 Predicted Genre:")
            st.markdown(f"**`{predicted_genre}`**")

            # === YouTube Search ===
            st.subheader("🎥 Top YouTube Results:")
            search_query = f"top {predicted_genre} songs"
            videosSearch = VideosSearch(search_query, limit=3)
            results = videosSearch.result()

            for i, video in enumerate(results['result'], 1):
                title = video['title']
                link = video['link']
                st.markdown(f"{i}. [{title}]({link})")

            # === Feature-Based Recommendations ===
            st.subheader("🔍 Similar Songs (Based on Features):")
            try:
                results = recommend_similar_songs(
                    file_path,
                    dataset_csv='content_based/features_dataset.csv',
                    top_n=5
                )
                if results:
                    for song, score in results:
                        st.write(f"🎵 {song} — Similarity: `{score:.4f}`")
                else:
                    st.warning("No similar songs found.")
            except Exception as e:
                st.error(f"❌ Error during recommendation: {e}")

        except Exception as e:
            st.error(f"❌ Error during genre prediction: {e}")
