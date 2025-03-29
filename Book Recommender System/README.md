# Book Recommendation System using KNN (Surprise Library)

This project builds a **collaborative filtering-based book recommendation system** using **KNNBasic** from the Surprise library. The dataset consists of user ratings for books from Goodreads.

## 📌 Features
- Loads and preprocesses the dataset
- Filters out invalid ratings (e.g., 0 ratings)
- Uses the `Surprise` library to build a **KNN-based recommendation model**
- Splits data into **training (80%)** and **testing (20%)**
- Predicts a user's rating for a given book

## 📂 Dataset
The dataset consists of:
- `user_id`: Unique identifier for users
- `book_id`: Unique identifier for books
- `rating`: User's rating (from 1 to 5)


