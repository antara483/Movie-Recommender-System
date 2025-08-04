import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

movies = pd.read_csv("movies_with_posters.csv")

def get_tmdb_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None
    return None

# Update missing posters
for i, row in movies.iterrows():
    if pd.isna(row['poster_url']) or row['poster_url'].strip() == "":
        poster = get_tmdb_poster(row['title'])
        if poster:
            movies.at[i, 'poster_url'] = poster

movies.to_csv("movies_with_posters.csv", index=False)
print("Updated CSV saved with posters.")
