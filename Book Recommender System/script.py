import pandas as pd
import codecademylib3
from surprise import Reader

book_ratings = pd.read_csv('goodreads_ratings.csv')
print(book_ratings.head())

#1. Print dataset size and examine column data types
print(book_ratings.head())
print(book_ratings.info())
#2. Distribution of ratings
print(book_ratings.rating.value_counts())
#3. Filter ratings that are out of range
book_ratings = book_ratings[book_ratings['rating']!= 0]

#4. Prepare data for surprise: build a Suprise reader object
from surprise import Reader
read = Reader(rating_scale=(1,5))
#5. Load `book_ratings` into a Surprise Dataset
from surprise import Dataset
dataset = Dataset.load_from_df(book_ratings[['user_id', 'book_id', 'rating']], read)
#6. Create a 80:20 train-test split and set the random state to 7
from surprise.model_selection import train_test_split
training_data, testing_data = train_test_split(dataset, random_state=7, test_size =.2)
#7. Use KNNBasice from Surprise to train a collaborative filter
from surprise import KNNBasic
book_recommender = KNNBasic()
book_recommender.fit(training_data)

#8. Evaluate the recommender system
from surprise import accuracy
pred = book_recommender.test(testing_data)
#9. Prediction on a user who gave the "The Three-Body Problem" a rating of 5
user_id = "8842281e1d1347389f2ab93d60773d4d"
book_id = 18007564
prediction = book_recommender.predict(user_id, book_id)
# Print the estimated rating
print(f"Predicted rating for user {user_id} on book {book_id}: {prediction.est}")
