# SmartFlix - Movie Recommendation System

This project is a movie recommendation system developed using Streamlit and various machine learning techniques. The system leverages collaborative filtering and matrix factorization methods to provide personalized movie recommendations.

## Data

The project uses data from the Netflix Prize dataset available on [Kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data/data). The dataset includes:
- `combined_data_1.txt`
- `combined_data_2.txt`
- `combined_data_3.txt`
- `combined_data_4.txt`
- `movie_titles.csv`

The `combined_data_{i}.txt` files contain user ratings for movies, where each file starts with a movie ID followed by ratings from different users. The `movie_titles.csv` file contains the movie ID, year of release, and title.

## Project Components

### dep.py
The `dep.py` file includes the following key functions:

- `get_ratings(predictions)`: Extracts actual and predicted ratings from model predictions.
- `get_errors(predictions, print_them=False)`: Calculates RMSE and MAPE from predictions.
- `run_surprise(algo, trainset, testset, verbose=True)`: Trains and evaluates a Surprise model.

### movie.py
The `movie.py` file sets up the Streamlit app and includes the following main components:

- Streamlit configuration and header setup.
- User input for movie ID.
- Functionality to fetch and display movie recommendations.

### movie-recommender.ipynb
The `movie-recommender.ipynb` notebook includes:

- Data overview and exploration.
- Preliminary analysis of the Netflix Prize dataset.


### Collaborative Filtering
Based on the combined effect of user-user and movie-user collaborative filtering.

## Overall Implementation of the Movie Recommender System is Divided into 3 Benchmarks

### 1. Part 1
- Predicting Top 10 movies for a given movie.
- Cross-checking the above list via Google’s Gemini.

### 2. Part 2
- Validating that matrix-factorization models are superior to classic nearest-neighbor techniques for producing product recommendations.
- Minimizing the difference between predicted and actual ratings via two performance metrics: RMSE and MAPE.


### Part 3: SVD Model Comparison

- Using the SVD matrix-factorization model for predicting Top 10 unwatched movies for a given user and a given watched movie.
- Comparing my SVD model with the current state-of-the-art algorithm.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Udbhav514/SmartFlix.git
    cd movie-recommender
    ```

2. Create a `.env` file to store environment variables (if needed):
    ```bash
    touch .env
    ```

## Usage

To start the Streamlit app, run:
```bash
streamlit run dep.py

