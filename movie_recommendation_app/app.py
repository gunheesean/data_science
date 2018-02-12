from flask import Flask, render_template,request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import pandas as pd
import random

app = Flask(__name__)
cur_dir = os.path.dirname(__file__)

########################
# db
########################
db = os.path.join(cur_dir, 'DB.sqlite')

########################
# load pickle
########################
smd = pickle.load(open(os.path.join(cur_dir,
                                'pkl_objects',
                                'smd.pkl'), 'rb'))
indices = pickle.load(open(os.path.join(cur_dir,
                                'pkl_objects',
                                'indices.pkl'), 'rb'))
id_map = pickle.load(open(os.path.join(cur_dir,
                                'pkl_objects',
                                'id_map.pkl'), 'rb'))
cosine_sim = pickle.load(open(os.path.join(cur_dir,
                                'pkl_objects',
                                'cosine_sim.pkl'), 'rb'))
svd = pickle.load(open(os.path.join(cur_dir,
                                'pkl_objects',
                                'svd.pkl'), 'rb'))
knn = pickle.load(open(os.path.join(cur_dir,
                                'pkl_objects',
                                'knn.pkl'), 'rb'))

#########################
# functions
#########################
def sqlite_entry(path, a_movie, rate, r_movie1, r_movie2, r_movie3):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO movie_db (ask_movie, rate, rec_movie1, rec_movie2, rec_movie3, date)"\
    " VALUES (?, ?, ?, ?, ?, DATETIME('now'))", (a_movie, rate, r_movie1, r_movie2, r_movie3))
    conn.commit()
    conn.close()

# recommend 3 similar users
def recommend(rate, title):
    indices_map = id_map.set_index('id')
    idx = indices[title]
    movie_id = int(id_map.loc[title]['id'])

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # top 5 movies
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    # find user
    rating_vector = np.zeros((1,9066)).flatten()
    rating_vector[idx] = rate
    userId = recommend_user(rating_vector)
    # predict
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.drop('id', axis=1)
    movies = movies.sort_values('est', ascending=False)
    return movies.title.iloc[:3].values

# get one the most similar user
def recommend_user(rating_vec):
    # reshape vector to fit knn model
    rating_vector = rating_vec.reshape(1,-1)
    users = knn.kneighbors(rating_vector, return_distance=False)
    return users.flatten()[-1]


#########################
# Flask
#########################
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=3)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        rate = request.form['rate']
        ask_movie = request.form['moviereview']
        rec_movies = recommend(rate, ask_movie)
        movie1, movie2, movie3 = rec_movies[0], rec_movies[1], rec_movies[2]
        return render_template('results.html',
                                ask=ask_movie,
                                rating=rate,
                                content1=movie1,
                                content2=movie2,
                                content3=movie3)
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    ask = request.form['ask']
    rating = request.form['rating']
    movie1 = request.form['movie1']
    movie2 = request.form['movie2']
    movie3 = request.form['movie3']

    sqlite_entry(db, ask, rating, movie1, movie2, movie3)
    return render_template('thanks.html')

if __name__ == '__main__':
	app.run(debug=True)
