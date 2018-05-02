/**
 * @file readfile.hpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Reads in the file and saves it in several arrays in our data object
 */

#include <string>
#include <tuple>

using namespace std;

class Data 
{
private:
    tuple<int, int, int, int> *array_1;
    tuple<int, int, int, int> *array_2;
    tuple<int, int, int, int> *array_3;
    tuple<int, int, int, int> *array_4;
    tuple<int, int, int> *array_5;
public:
    Data(); // Constructor
    ~Data(); // Destructor
    tuple<int, int, int, int> *getArray(int idx); // Returns array_idx
    tuple<int, int, int> *getQual();
};
