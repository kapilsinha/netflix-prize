ARRAY_1_SIZE = 94362233 # Training data
ARRAY_2_SIZE = 1965045 # Validation data
ARRAY_3_SIZE = 1964391 # Hidden data
ARRAY_4_SIZE = 1374739 # Probe into test set
ARRAY_5_SIZE = 2749898 # Qual data

filename = "PMF_probe_qual.txt"
file = open(filename)
preds = []
for line in file.readlines():
    preds.append(line.strip().split(" ")[-1])
file.close()

probe = preds[:ARRAY_4_SIZE]
qual = preds[ARRAY_4_SIZE:]

if len(preds) != ARRAY_4_SIZE + ARRAY_5_SIZE:
    print("Error! Make sure you removed the header (first few lines) that are not data")

file = open("PMF_probe.txt", "w")
file.writelines(["%s\n" % item for item in probe])

file = open("PMF_qual.txt", "w")
file.writelines(["%s\n" % item for item in qual])
