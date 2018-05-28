import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras import optimizers
LEN_PROBE = 1374739
probe = "mu_probe.csv"
qual = "mu_qual.csv"

# Models

# Add filename of probe predictions to list below
probe_models = ["Time_SVD_probe_preds_200_factors.txt", "SVD++_probe_preds_100_factors.txt", "RBM_probe.txt", "PMF_probe.txt"]
qual_models = ["Time_SVD_preds_100_factors_probetrained.txt", "SVD++_preds_100_factors_probetrained.txt", "RBM_qual.txt", "PMF_qual.txt"]

if len(probe_models) != len(qual_models):
    raise ValueError("Must have qual and probe predictions for all models")

# Loop over the models to read in the training data
X = []
for model in probe_models:
    data = [line for line in open(model)]
    if len(data) < LEN_PROBE:
        raise ValueError("Length of input should be equal to length of probe dataset")
    X.append(data)
X = np.array(X).T

# Read in probe data (Training data)
y = np.array([int(line.split(",")[3]) for line in open(probe)])

# Read in model qual data
test = []
for model in qual_models:
    data = [line for line in open(model)]
    test.append(data)
test = np.array(test).T

# Model
model = Sequential()
model.add(Dense(15, input_dim=len(probe_models)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
model.summary()
model.fit(X, y, epochs=15, batch_size=256)

# Predict on qual set
print("Writing predictions")
preds = model.predict(test)
# Cap the ratings
for i in range(len(preds)):
  if preds[i] < 1:
      preds[i] = 1
  elif preds[i] > 5:
      preds[i] = 5
file = open("blended.txt", "w")
file.writelines(["%s\n" % item[0] for item in preds])