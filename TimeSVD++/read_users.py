import numpy as np

data = np.genfromtxt("mu_val.csv", delimiter=",")
# Remove the date
data = np.delete(data, 2, 1)
data = data[1:]

users = [[0, 0]] 
for i in range(458293):
	users.append([0, 0])

for i in data:
	user = int(i[0])
	movie = i[1]
	rating = i[2]

	users[user][0] += 1
	users[user][1] += rating

	if user == 456345:
		print("user found")

users[0] = ["Num Ratings", "Sum Ratings"]


print("Saving to file")
# Save to a file
file = open("rating_info.txt", "w")
for item in users:
	file.write("%s\n" % item)

