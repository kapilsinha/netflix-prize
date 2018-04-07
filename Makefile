CC = g++
LD = g++
CPPFLAGS = -std=c++11 -g -Wall -pedantic
LDFLAGS =

# Add any extra .cpp files to this line to
# have them compiled automatically.
SRCS = readfile.cpp matrix_factorization.cpp

OBJS = $(SRCS:.cpp=.o)

all: readfile matrix_factorization

readfile: $(OBJS)
	g++ -o $@ $^ $(LDFLAGS)

%.o : %.cpp
	$(CC) -c $(CPPFLAGS) $< -o $@

clean :
	rm -f readfile matrix_factorization *.o
