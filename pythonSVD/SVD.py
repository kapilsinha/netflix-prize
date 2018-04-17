from surprise import SVD
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.reader import Reader
from surprise.dataset import Dataset, DatasetAutoFolds
import pandas as pd
import progressbar

K = 20

# We'll use the famous SVD algorithm.
algo = SVD(n_factors=K)

# Build dataset from csv file
print("Reading data ...")
data = pd.read_csv("mu_train.csv")
print("Convert to dataset ...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[["User Number", "Movie Number", "Rating"]], reader)
print("Building training set ...")
train = DatasetAutoFolds.build_full_trainset(data)

print("Training ...")

# Fit the data
algo.fit(train)


# Write the predictions
print("Writing predictions ...")
preds = []
x = 0
file = "mu/qual.dta"
with open(file) as f:
    for line in f:
        if x % 100000 == 0:
            print(x)
        x += 1
        line = line.split()
        user = int(line[0])
        item = int(line[1])
        print(algo.predict(user, item))
        preds.append(algo.predict(user, item))

# Write to a text file
file = open("predictions.txt", "w")
x = 0
for item in preds:
    x += 1
    file.write("%g\n" % item.est)
print("Done. %d entries written." % x)