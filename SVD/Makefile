CC = g++
LD = g++
CPPFLAGS = -std=c++11 -g -Wall -pedantic
#LDFLAGS =


# Add any extra .cpp files to this line to
# have them compiled automatically.
SRCS = readfile.cpp matrix_factorization.cpp predict.cpp

#OBJS = $(SRCS:.cpp=.o)

all: predict

readfile: readfile.o
	$(CC) $(CPPFLAGS) -o readfile readfile.o

matrix_factorization: readfile.o matrix_factorization.o
	$(CC) $(CPPFLAGS) -o matrix_factorization readfile.o matrix_factorization.o

predict: matrix_factorization.o readfile.o predict.o 
	$(CC) $(CPPFLAGS) -o predict matrix_factorization.o readfile.o predict.o

readfile.o: readfile.cpp readfile.hpp
	$(CC) $(CPPFLAGS) -c readfile.cpp

matrix_factorization.o: matrix_factorization.cpp matrix_factorization.hpp readfile.hpp
	$(CC) $(CPPFLAGS) -c matrix_factorization.cpp

predict.o: predict.cpp predict.hpp readfile.hpp matrix_factorization.hpp 
	$(CC) $(CPPFLAGS) -c predict.cpp

clean :
	rm -f readfile matrix_factorization predict *.o
