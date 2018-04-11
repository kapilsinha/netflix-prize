import pandas as pd
import numpy as np

# Make sure your file paths are right when you are running this file

# Function to parse and separate the mu data
def get_mu_data(path, inds):
    # Read data into a pandas dataframe
    mu_data = pd.read_csv(mu_path, sep=' ', header=None)
    mu_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]

    # Read indices
    mu_idx = pd.read_table(mu_ind_path, header=None)
    mu_idx.columns = ["Index"]

    # Get the data for train, val, hidden, probe, and qual
    dataLists = []
    for i in range(1, 6):
        val = mu_idx.loc[mu_idx["Index"] == i]
        val = val.index.tolist()
        dataLists.append(mu_data.loc[val])

    # Output all the data to csv files
    dataLists[0].to_csv("mu_train.csv")
    dataLists[1].to_csv("mu_val.csv")
    dataLists[2].to_csv("mu_hidden.csv")
    dataLists[3].to_csv("mu_probe.csv")
    dataLists[4].to_csv("mu_qual.csv")

# Function to parse and separate the um data
def get_um_data(path, inds):
    # Read data into a pandas dataframe
    um_data = pd.read_csv(um_path, sep=' ', header=None)
    um_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]

    # Read indices
    um_idx = pd.read_table(um_ind_path, header=None)
    um_idx.columns = ["Index"]

    # Get the data for train, val, hidden, probe, and qual
    dataLists = []

    for i in range(1, 6):
        val = um_idx.loc[um_idx["Index"] == i]
        val = val.index.tolist()
        dataLists.append(um_data.loc[val])

    # Output all the data to csv files
    dataLists[0].to_csv("um_train.csv")
    dataLists[1].to_csv("um_val.csv")
    dataLists[2].to_csv("um_hidden.csv")
    dataLists[3].to_csv("um_probe.csv")
    dataLists[4].to_csv("um_qual.csv")

# Get the file paths
mu_path = "mu/all.dta"
mu_ind_path = "mu/all.idx"

um_path = "um/all.dta"
um_ind_path = "um/all.idx"

get_mu_data(mu_path, mu_ind_path)
get_um_data(um_path, um_ind_path)
