

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
import time
import urllib.parse
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import io
import textwrap
import os

load_dotenv()  # Load variables from .env

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Configuration

MAX_RETRIES = 2
RETRY_DELAY = 1.0
TIMEOUT = 10
CACHE_DIR = "poster_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Streamlit setup
st.set_page_config(page_title="Movie Recommender System", layout="wide")
st.title('🎬 Movie Recommender System')

@st.cache_data
def load_data():
    movies = pd.read_csv("movies_with_posters.csv")
    movies.dropna(subset=["overview"], inplace=True)
    movies['poster_url'] = movies['poster_url'].fillna("")
    movies.reset_index(drop=True, inplace=True)
    return movies

movies = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

def create_text_poster(movie_title):
    """Generate a custom poster with movie title text"""
    img = Image.new('RGB', (500, 750), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()

    wrapped_text = textwrap.fill(movie_title, width=15)
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (500 - text_width) / 2
    y = (750 - text_height) / 2
    draw.multiline_text((x, y), wrapped_text, font=font, fill=(50, 50, 50), align="center")

    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def save_to_cache(poster_url, movie_title):
    """Save poster to cache"""
    try:
        response = requests.get(poster_url, stream=True, timeout=TIMEOUT)
        if response.status_code == 200:
            safe_title = "".join(c if c.isalnum() else "_" for c in movie_title)
            cache_path = os.path.join(CACHE_DIR, f"{safe_title}.jpg")
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return cache_path
    except:
        pass
    return None

def get_cached_poster(movie_title):
    """Check cache"""
    safe_title = "".join(c if c.isalnum() else "_" for c in movie_title)
    cache_path = os.path.join(CACHE_DIR, f"{safe_title}.jpg")
    return cache_path if os.path.exists(cache_path) else None

def fetch_poster_silently(movie_title):
    """Fetch from TMDB, return path or None"""
    cached_path = get_cached_poster(movie_title)
    if cached_path:
        return cached_path

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    title_variations = [
        movie_title,
        movie_title.split('(')[0].strip(),
        ' '.join(movie_title.split()[:4]),
        ' '.join(movie_title.split()[:3]),
    ]

    for variation in title_variations:
        params = {
            "api_key": TMDB_API_KEY,
            "query": variation,
            "language": "en-US"
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params=params,
                    headers=headers,
                    timeout=TIMEOUT
                )

                if response.status_code == 429:
                    time.sleep(10)
                    continue

                if response.status_code == 200:
                    data = response.json()
                    if data.get("results"):
                        for result in data["results"]:
                            if result.get("poster_path"):
                                poster_url = f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                                cached_path = save_to_cache(poster_url, movie_title)
                                return cached_path or poster_url

            except:
                pass

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return None

def get_poster(movie_title, local_poster_path=""):
    """Always return a valid poster (path, URL, or BytesIO)"""
    if local_poster_path and isinstance(local_poster_path, str):
        if local_poster_path.startswith("http"):
            return local_poster_path
        elif local_poster_path and not local_poster_path.startswith("/"):
            return f"https://image.tmdb.org/t/p/w500{local_poster_path}"

    poster = fetch_poster_silently(movie_title)
    if poster:
        return poster

    return create_text_poster(movie_title)

def recommend(title, top_n=5):
    if title not in movies['title'].values:
        return [], []

    idx = movies[movies['title'] == title].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    titles = movies['title'].iloc[movie_indices].tolist()
    posters = [get_poster(movies.iloc[i]['title'], movies.iloc[i]['poster_url']) for i in movie_indices]
    return titles, posters

# UI
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie to get recommendations:", movie_list)

if st.button("Recommend", type="primary"):
    with st.spinner('Loading recommendations...'):
        names, posters = recommend(selected_movie)

    if names:
        st.subheader("Top Recommendations:")
        cols = st.columns(len(names))

        for i, col in enumerate(cols):
            with col:
                poster = posters[i]
                try:
                    if isinstance(poster, str) and os.path.exists(poster):
                        st.image(poster, use_container_width=True)
                    elif isinstance(poster, str):
                        st.image(poster, use_container_width=True)
                    else:  # BytesIO object
                        st.image(poster, use_container_width=True)

                    st.caption(names[i])
                except:
                    st.image(create_text_poster(names[i]), use_container_width=True)
                    st.caption(names[i])
    else:
        st.error("No recommendations found. Please try another movie.")
