import os
import requests

def search_youtube(query, max_results=5):
    api_key = os.getenv("YT_API_KEY")
    if not api_key:
        print("❌ YouTube API key not found. Make sure it's set using: set YT_API_KEY=YOUR_KEY")
        return []

    search_url = "https://www.googleapis.com/youtube/v3/search"
    
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': max_results,
        'key': api_key
    }

    response = requests.get(search_url, params=params)

    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code} - {response.text}")
        return []

    results = response.json().get('items', [])
    videos = []

    for item in results:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append((title, url))

    return videos


# Example usage:
if __name__ == "__main__":
    genre = input("🎧 Enter predicted genre or mood (e.g., 'rock', 'lofi', 'classical'): ")
    query = f"top {genre} songs"
    print(f"\n🔍 Searching YouTube for: {query}\n")

    songs = search_youtube(query)
    if not songs:
        print("❌ No songs found.")
    else:
        print("🎵 Here are some real YouTube songs:")
        for title, url in songs:
            print(f"- {title}\n  {url}")
