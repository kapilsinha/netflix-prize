import pandas as pd
import numpy as np

# Make python lists
mu_path = "mu/all.dta"
mu_ind_path = "mu/all.idx"

um_path = "um/all.dta"
um_ind_path = "um/all.idx"

fo = open("", "rw+")

with open(fname) as f:
    content = f.readlines()

mu_data = pd.read_csv(mu_path, sep=' ', header=None)
mu_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]
mu_idx = pd.read_table(mu_ind_path, header=None)
mu_idx.columns = ["Index"]

# Get the data for train, val, hidden, probe, and qual

dataLists = []

for i in range(1, 6):
    val = mu_idx.loc[mu_idx["Index"] == i]
    val = val.index.tolist()
    dataLists.append(mu_data.loc[val])

dataLists[0].to_csv("data/mu_train.csv")
dataLists[1].to_csv("data/mu_val.csv")
dataLists[2].to_csv("data/mu_hidden.csv")
dataLists[3].to_csv("data/mu_probe.csv")
dataLists[4].to_csv("data/mu_qual.csv")


# UM data

um_data = pd.read_csv(um_path, sep=' ', header=None)
um_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]
um_idx = pd.read_table(um_ind_path, header=None)
um_idx.columns = ["Index"]

dataLists = []

for i in range(1, 6):
    val = um_idx.loc[um_idx["Index"] == i]
    val = val.index.tolist()
    dataLists.append(um_data.loc[val])

dataLists[0].to_csv("data/um_train.csv")
dataLists[1].to_csv("data/um_val.csv")
dataLists[2].to_csv("data/um_hidden.csv")
dataLists[3].to_csv("data/um_probe.csv")
dataLists[4].to_csv("data/um_qual.csv")
