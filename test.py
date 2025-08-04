

import pandas as pd
import requests
import urllib.parse
import time
import os
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DATA_FILE = "dataset.csv"
OUTPUT_FILE = "movies_with_posters.csv"

# Load dataset
movies = pd.read_csv(DATA_FILE)
movies['overview'] = movies['overview'].fillna('')
movies['genre'] = movies['genre'].fillna('')

# If posters already fetched before, load that and continue
if os.path.exists(OUTPUT_FILE):
    movies_with_posters = pd.read_csv(OUTPUT_FILE)
    if 'poster_url' not in movies_with_posters.columns:
        movies_with_posters['poster_url'] = None
else:
    movies_with_posters = movies.copy()
    movies_with_posters['poster_url'] = None

# Poster fetching function with retry
def fetch_poster(title, retries=3):
    for attempt in range(retries):
        try:
            query = urllib.parse.quote(title)
            url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            return None
        except Exception as e:
            print(f"Attempt {attempt+1} failed for '{title}': {e}")
            time.sleep(2 ** attempt)
    return None

# Loop through and fetch only missing posters
for idx, row in movies_with_posters.iterrows():
    if pd.isna(row['poster_url']) or row['poster_url'] == "None":
        title = row['title']
        print(f"Fetching: {title}")
        poster_url = fetch_poster(title)
        movies_with_posters.at[idx, 'poster_url'] = poster_url
        print(f"{title} -> {poster_url}")
        time.sleep(2)  # respect rate limits

        # Save progress every 10 movies
        if idx % 10 == 0:
            movies_with_posters.to_csv(OUTPUT_FILE, index=False)
            print("Progress saved.")

# Final save
movies_with_posters.to_csv(OUTPUT_FILE, index=False)
print("✅ Done! All posters fetched and saved.")
