from flask import Flask
from flask import request
from load_data import data
from recommender import algo

app = Flask(__name__)

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
  trainingSet = data.build_full_trainset()

  algo.fit(trainingSet)
  print (request.data)
  return "Hello World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)