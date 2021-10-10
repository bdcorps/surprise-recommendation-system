from load_data import data
from recommender import algo
from surprise import dump as Dump

trainingSet = data.build_full_trainset()

algo.fit(trainingSet)

Dump.dump("model.pkl", None, algo, 0)

# prediction = algo.predict('E', 2)

uid = str(1)  # raw user id (as in the ratings file). They are **strings**!
iid = str(200)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
prediction = algo.predict(uid, iid, r_ui=4, verbose=True)

print (prediction)