import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_genres(text):
    """
    Function to extract genre names from a string representation of a 
    dictionary list
    """
    # Extracting values of 'name' keys
    names = re.findall(r"'name': '([^']+)'", text)  
    return ", ".join(names)  # Returning as a comma-separated string


def clean_keywords(text):
    """
    Function to extract keyword names from a string representation of a 
    dictionary list
    """
    # Extracting values of 'name' keys
    names = re.findall(r"'name': '([^']+)'", text)  
    return ", ".join(names)


def read_metadata():
    """
    Function to read and filter movie metadata
    """
    metadata = pd.read_csv("data/movies_metadata.csv")[
        ["id", "title", "genres", "overview", "vote_average", "vote_count"]
    ]
    metadata = metadata[
        metadata["vote_count"] > 1000
    ]  # Filtering movies with more than 1000 votes
    metadata = metadata.sort_values(
        "vote_average", ascending=False
    )  # Sorting by vote average
    metadata = metadata.iloc[:500]  # Selecting top 500 movies
    metadata["id"] = pd.to_numeric(
        metadata["id"], errors="coerce"
    )  # Converting id to numeric, handling errors
    # Dropping rows with NaN values in 'id'
    metadata = metadata.dropna(subset=["id"])  
    return metadata


def read_keywords():
    """
    Function to read and deduplicate keyword data
    """
    keywords = pd.read_csv("data/keywords.csv")
    keywords = keywords.drop_duplicates(
        subset=["id"], keep="first"
    )  # Removing duplicate entries by 'id'
    return keywords


def combine_datasets(metadata, keywords):
    """
    Function to combine metadata and keywords datasets
    """
    dataset = pd.merge(metadata, keywords, on="id", how="inner")[
        ["title", "genres", "keywords", "overview"]
    ]
    # Cleaning genres
    dataset["genres"] = dataset["genres"].apply(clean_genres)  
    # Cleaning keywords
    dataset["keywords"] = dataset["keywords"].apply(clean_keywords)  
    dataset["description"] = (
        dataset["genres"].astype(str)
        + ". "
        + dataset["keywords"].astype(str)
        + ". "
        + dataset["overview"].astype(str)
    )  # Creating a description column
    dataset = dataset[["title", "description"]]
    dataset.to_csv(
        "data/processed_data.csv", index=False
    )  # Saving processed data to CSV
    return dataset


def recommend_movies(user_input, top_n=5):
    """
    Function to recommend movies based on user input
    """
    user_tfidf = tfidf_vectorizer.transform(
        [user_input]
    )  # Transform user input into TF-IDF vector
    similarities = cosine_similarity(
        user_tfidf, tfidf_matrix
    ).flatten()  # Compute cosine similarity
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top N indices
    movies = list(dataset.iloc[top_indices]["title"])  # Retrieve movie titles

    print("Here's a list of movies tailored to your interests:")
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie}")


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = pd.read_csv("data/processed_data.csv")
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset["description"])
    print("Dataset loading complete!")

    while True:
        prompt = input(
            'Please describe the movies your like, or type "Exit" to leave: '
        )
        if prompt.strip() == "Exit":
            break
        else:
            recommend_movies(prompt)
