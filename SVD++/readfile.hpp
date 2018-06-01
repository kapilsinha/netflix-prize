/**
 * @file readfile.hpp
 * @date 04/06/18
 *
 * @brief Reads in the file and saves it in several arrays in our data object
 */

#include <string>
#include <tuple>
#include <vector>

using namespace std;

class Data
{
private:
    tuple<int, int, int, int> *array_1;
    tuple<int, int, int, int> *array_2;
    tuple<int, int, int, int> *array_3;
    tuple<int, int, int, int> *array_4;
    tuple<int, int, int> *array_5;
    vector<tuple<int, int, int>> *get_user_data(tuple<int, int, int, int> *arr, int size);

public:
    Data(); // Constructor
    ~Data(); // Destructor
    tuple<int, int, int, int> *getArray(int idx); // Returns array_idx
    tuple<int, int, int> *getQual();
    vector<tuple<int, int, int>> *format_user_data(int idx);
    vector<tuple<int, int, int>> *get_all_data(tuple<int, int, int, int> *arr1, int size1,
        tuple<int, int, int, int> *arr2, int size2, 
        tuple<int, int, int, int> *arr3, int size3, 
        tuple<int, int, int, int> *arr4, int size4);
};
