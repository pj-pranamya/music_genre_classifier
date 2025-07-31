import os
import json

# Path to your dataset folder
genre_dir = "genres"

# Get list of genre folder names and sort them to match training order
genres = sorted(os.listdir(genre_dir))

# Create label map: {0: "classical", 1: "metal", ...}
label_map = {i: genre for i, genre in enumerate(genres)}

# Save as JSON
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("✅ label_map.json created successfully!")
