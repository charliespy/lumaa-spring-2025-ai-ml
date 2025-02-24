<!-- #region -->
# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

## Overview

This project implements a **content-based recommendation system** that suggests **movies** based on a user’s short text description of preferences. The system uses **TF-IDF vectorization** and **cosine similarity** to compare user input with a dataset of movie descriptions and recommend the most relevant movies.

## Dataset

- The dataset consists of movie metadata, including **title, genres, keywords, and overview**.
- Data is sourced from ["The Movies Dataset"](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=credits.csv) on Kaggle.
- Processed data is stored in `data/processed_data.csv` for quick access.

## Setup & Installation

### Requirements
Ensure you have Python installed (preferably Python 3.9+). Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Code

To start the recommendation system, run:
```bash
python recommender.py
```

The system will prompt you to enter a **movie description**. Type a sentence describing the kind of movies you like, and it will return the top 5 recommended movies.

To exit, type:
```bash
Exit
```

## Implementation Details

### Steps:
1. **Load Dataset**: Reads movie metadata and keyword files.
2. **Preprocess Data**: Cleans genre and keyword information.
3. **Feature Extraction**: Uses **TF-IDF vectorization** to convert text into numerical representations.
4. **Similarity Calculation**: Uses **cosine similarity** to compare user input with movie descriptions.
5. **Recommendation**: Returns the top 5 most relevant movies.

### Example Usage
#### Input:
```
I love thrilling action movies set in space, with a comedic twist.
```
#### Output:
```
Here's a list of movies tailored to your interests:
1. Rogue One: A Star Wars Story
2. Avatar
3. Alien
4. Gravity
5. Guardians of the Galaxy Vol. 2
```

## Deliverables
1. **Code**: Python scripts implementing the recommendation system.
2. **README.md**: This file, providing setup and usage instructions.
3. **Demo Video**: A short screen recording demonstrating functionality linked [here](www.youtube.com).
4. **Salary Expectation**: 2000 - 4000 per month.
