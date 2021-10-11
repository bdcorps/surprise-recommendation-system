from flask import Flask
from flask import request
from load_data import data
from recommender import algo
from flask_cors import CORS
from flask import render_template

import pandas as pd
from surprise import Dataset
from surprise import Reader
from flask import jsonify

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/query/', methods=['POST'])
def query():
  print (request.values, request.get_json())



  query_users = request.values.getlist('users[]')
  query_items = request.values.getlist('items[]')
  query_ratings = request.values.getlist('ratings[]')
  query_items_to_predict = request.values.getlist('items_to_predict[]')

  items =['Titanic', 'The Karate Kid','Pulp Fiction','Back to the Future', 'Titanic', 'The Karate Kid','Pulp Fiction','Back to the Future','Titanic', 'The Karate Kid','Pulp Fiction','Back to the Future','Titanic', 'The Karate Kid','Pulp Fiction','Back to the Future']
  users = ['A', 'A','A', 'A', 'B', 'B','B', 'B', 'C', 'C','C', 'C', 'D', 'D','D', 'D']
  ratings = [1,2,3,2,4,5,3,2,1,2,3,5,1,2,4,5]

  users = users + query_users
  items = items+ query_items
  ratings = ratings+ query_ratings

  ratings_dict = {
      "item": items,
      "user": users,
      "rating": ratings
  }

  df = pd.DataFrame(ratings_dict)
  reader = Reader(rating_scale=(1, 5))

  # Loads Pandas dataframe
  data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
  trainingSet = data.build_full_trainset()

  algo.fit(trainingSet)

  print( users,
  items,
  ratings )

  predictions=[]
  for item in query_items_to_predict:
    print(algo.predict(query_users[0], item, r_ui=4, verbose=True))
    predictions.append(algo.predict(query_users[0], item, r_ui=4, verbose=True))

  return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)