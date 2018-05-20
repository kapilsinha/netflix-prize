CC = g++
LD = g++
# CPPFLAGS = -std=c++11 -g -Wall -pedantic
CPPFLAGS = -std=c++11 -g -Wall -pedantic -fPIC

# Add any extra .cpp files to this line to
# have them compiled automatically.
SRCS = readfile.cpp svdplusplus.cpp predict.cpp

#OBJS = $(SRCS:.cpp=.o)

all: predict svdplusplus readfile

readfile: readfile.o
	# $(CC) $(CPPFLAGS) -o readfile readfile.o
	$(CC) -shared -Wl,-soname,libreadfile.so -o libreadfile.so readfile.o

svdplusplus: readfile.o svdplusplus.o
	# $(CC) $(CPPFLAGS) -o svdplusplus readfile.o svdplusplus.o
	$(CC) -shared -Wl,-soname,libsvdplusplus.so -o libsvdplusplus.so readfile.o svdplusplus.o

predict: svdplusplus.o readfile.o predict.o
	$(CC) $(CPPFLAGS) -o predict svdplusplus.o readfile.o predict.o
	$(CC) -shared -Wl,-soname,libpredict.so -o libpredict.so svdplusplus.o readfile.o predict.o

readfile.o: readfile.cpp readfile.hpp
	$(CC) $(CPPFLAGS) -c readfile.cpp

svdplusplus.o: svdplusplus.cpp svdplusplus.hpp readfile.hpp
	$(CC) $(CPPFLAGS) -c svdplusplus.cpp

predict.o: predict.cpp predict.hpp readfile.hpp svdplusplus.hpp
	$(CC) $(CPPFLAGS) -c predict.cpp

clean :
	rm -f readfile svdplusplus predict *.o *.so
