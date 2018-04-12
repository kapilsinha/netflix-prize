from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd

K = 20

# Function to write predictions to file 
def write_preds(svd, file):
    preds = []
    with open(file) as f:
        for line in f:
            line = line.split()
            preds.append(svd.predict(line[0], line[1]))
    # TODO: Write preds to a file

# We'll use the famous SVD algorithm.
algo = SVD(n_factors=K)

# Build dataset from csv file
reader = Reader(line_format="user item timestamp rating")
data = pd.read_csv("mu_train.csv")
data = Dataset.load_from_df(data, reader)
train = DatasetAutoFolds.build_full_trainset(data)

# Fit the data
algo.fit(train)

