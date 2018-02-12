
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


ratings = pd.read_csv('the-movies-dataset/ratings_small.csv')
ratings.drop('timestamp', axis=1, inplace=True)

# ratings.shape, movies.shape --> ((100004, 3), (45466, 24))
# pivot dataframe and all nan values fill with 0.
rating_matrix = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating', fill_value=0)
# data frame to np.ndarry
rating_matrix = rating_matrix.as_matrix()
# rating_matrix.shape -->> (671, 9066)

#### get 5 top similar users by k-NN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(rating_matrix)
