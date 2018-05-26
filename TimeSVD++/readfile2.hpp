/**
 * @file readfile.hpp
 * @author Kapil Sinha
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
    tuple<int, int, int, int> *train;
    tuple<int, int, int> *array_5;
    vector<tuple<int, int, int>> *get_user_data(tuple<int, int, int, int> *arr, int size);

public:
    Data(); // Constructor
    ~Data(); // Destructor
    tuple<int, int, int, int> *getTrain(); // Returns array_idx
    tuple<int, int, int> *getQual();
    vector<tuple<int, int, int>> *format_user_data(int idx);
};
