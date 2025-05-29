import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movie_data = pd.read_csv('movies.csv')

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Fill missing values
for feature in selected_features:
    movie_data[feature] = movie_data[feature].fillna('')

# Combine features into a single string
movie_data['combined_features'] = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + movie_data['tagline'] + ' ' + movie_data['cast'] + ' ' + movie_data['director']

# Convert text data into numerical vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(movie_data['combined_features'])

# Compute cosine similarity
similarity = cosine_similarity(feature_vector)

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Enter a movie name to get similar recommendations.")

# User input for movie search
movie_name = st.text_input("Enter your favorite movie:", "")

if st.button("Find Recommendations"):
    if movie_name.strip():
        # Find closest matching movie
        list_of_all_titles = movie_data['title'].tolist()
        close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)

        if close_match:
            closest_movie = close_match[0]
            st.write(f"Closest match found: **{closest_movie}**")

            # Get the index of the movie
            movie_index = movie_data[movie_data.title == closest_movie].index[0]

            # Get similarity scores and sort them
            similarity_scores = list(enumerate(similarity[movie_index]))
            sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            # Display top 5 recommendations
            st.subheader("Top 10 Recommended Movies:")
            for i in range(1, 11):  # Skipping index 0 as it is the input movie itself
                index = sorted_movies[i][0]
                st.write(f"{i}. {movie_data.iloc[index]['title']}")
        else:
            st.write("No close match found. Please try another movie.")
    else:
        st.write("Please enter a movie name.")

