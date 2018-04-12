from surprise import SVD
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.reader import Reader
from surprise.dataset import Dataset, DatasetAutoFolds
import pandas as pd

K = 20

# Function to write predictions to file 
def write_preds(svd, file):
    preds = []
    with open(file) as f:
        for line in f:
            line = line.split()
            preds.append(svd.predict(line[0], line[1]))
    # TODO: Check if txt files are okay
    file = "predictions.txt"
    file.writelines(["%s\n" % item  for item in list])

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

write_preds(algo, "mu/qual.dta")