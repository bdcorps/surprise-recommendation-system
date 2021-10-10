from surprise import dump as Dump

[_,algo]=Dump.load("model.pkl")

prediction = algo.predict('E', 2)

print (prediction)