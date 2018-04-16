CC = g++
LD = g++
CPPFLAGS = -std=c++11 -g -Wall -pedantic
#LDFLAGS =


# Add any extra .cpp files to this line to
# have them compiled automatically.
SRCS = readfile.cpp matrix_factorization.cpp

#OBJS = $(SRCS:.cpp=.o)

all: matrix_factorization

readfile: readfile.o
	$(CC) $(CPPFLAGS) -o readfile readfile.o

matrix_factorization: readfile.o matrix_factorization.o
	$(CC) $(CPPFLAGS) -o matrix_factorization readfile.o matrix_factorization.o

readfile.o: readfile.cpp readfile.hpp
	$(CC) $(CPPFLAGS) -c readfile.cpp

matrix_factorization.o: matrix_factorization.cpp matrix_factorization.hpp readfile.hpp
	$(CC) $(CPPFLAGS) -c matrix_factorization.cpp

#readfile: $(OBJS)
#	g++ -o $@ $^ $(LDFLAGS)

#%.o : %.cpp
#	$(CC) -c $(CPPFLAGS) $< -o $@

clean :
	rm -f readfile matrix_factorization *.o
