ARRAY_1_SIZE = 94362233 # Training data
ARRAY_2_SIZE = 1965045 # Validation data
ARRAY_3_SIZE = 1964391 # Hidden data
ARRAY_4_SIZE = 1374739 # Probe into test set
ARRAY_5_SIZE = 2749898 # Qual data
sizes = {"train": ARRAY_1_SIZE, "val": ARRAY_2_SIZE,
         "hidden": ARRAY_3_SIZE, "probe": ARRAY_4_SIZE,
         "qual": ARRAY_5_SIZE}

M = 458293 # Number of users
N = 17770 # Number of movies

names = ["train", "val", "hidden", "probe", "qual"]
for name in names:
    filename = "mu_" + name + ".csv"
    f = open(filename, "r")
    time_file = open("time_" + name + ".txt", "w")
    regular_file = open(name + ".txt", "w")

    time_file.write("%%MatrixMarket matrix coordinate real general\n")
    regular_file.write("%%MatrixMarket matrix coordinate real general\n")

    time_file.write(str(M) + " " + str(N) + " " + str(sizes[name]) + "\n")
    regular_file.write(str(M) + " " + str(N) + " " + str(sizes[name]) + "\n")

    f.readline() # discard first row of titles
    for line in f:
        line = line[:-1] # get rid of \n
        x = line.replace(",", " ")
        time_file.write(x + "\n")
        y = line.split(",")
        regular_file.write(y[0] + " " + y[1] + " " + y[3] + "\n")
    f.close()
    time_file.close()
    regular_file.close()
