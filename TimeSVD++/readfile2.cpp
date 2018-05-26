/**
 * @file readfile.cpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Reads in the file and saves it in several arrays in our data object
 */

#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include "readfile.hpp"

#define ARRAY_1_SIZE 94362233
#define ARRAY_2_SIZE 1965045
#define ARRAY_3_SIZE 1964391
#define ARRAY_4_SIZE 1374739
#define ARRAY_5_SIZE 2749898
#define NUM_USERS 458293

using namespace std;

/**
 * @brief Constructs a Data instance, which contains the data arrays instance.
 */
Data::Data(void)
{
    // Not sure if an array of tuples is the most efficient option but it is
    // simple - feel free to change it later
    train = new tuple<int, int, int, int>
        [ARRAY_1_SIZE + ARRAY_2_SIZE + ARRAY_3_SIZE + ARRAY_4_SIZE];
    array_5 = new tuple<int, int, int> [ARRAY_5_SIZE];

    int train_index = 0;
    int array_5_index = 0;

    ifstream infile("../mu/all.dta");
    ifstream idxfile("../mu/all.idx");

    while (!infile.eof() & !idxfile.eof() ) {
        int user, movie, date, rating;
        int idx;
        infile >> user >> movie >> date >> rating;
        idxfile >> idx;
        // Populate the arrays
        if (idx == 1 || idx == 2 || idx == 3 | idx == 4) {
            train[train_index] = make_tuple(user, movie, date, rating);
            train_index++;
        }
        else if (idx == 5) {
            array_5[array_5_index] = make_tuple(user - 1, movie - 1, date);
            array_5_index++;
        }
        else {
            throw "Unexpected index in all.idx";
        }
    }
}

/**
 * @brief Destructs a Data instance.
 */
Data::~Data()
{
    delete[] train;
    train = NULL;
    delete[] array_5;
    array_5 = NULL;
}

/**
 * @brief Returns array based on the parameter index (1, 2, 3, 4)
 *        Else throws error
 * Array 1 - base
 * Array 2 - valid
 * Array 3 - hidden
 * Array 4 - probe
 * @return array
 */
tuple<int, int, int, int> *Data::getTrain()
{
    return train;
}

/**
 * Get qual data
 */
tuple<int, int, int> *Data::getQual()
{
    return array_5;
}

/**
 * Used only for testing purposes (printing in main method)
 */
void print_tuple(tuple<int, int, int, int> tup) {
    cout << "(" << get<0>(tup) << " " << get<1>(tup) << " "
         << get<2>(tup) << " " << get<3>(tup) << ")" << endl;
}

vector<tuple<int, int, int>> *Data::get_user_data(tuple<int, int, int, int> *arr, int size){
    vector<tuple<int, int, int>> *user_array = new vector<tuple<int, int, int>> [NUM_USERS];

    for (int i = 0; i < size; i++){
        tuple<int, int, int, int> info = arr[i];
        int user = get<0>(info);

        // Information given as movie, date, rating
        tuple<int, int, int> store_info = make_tuple(get<1>(info) - 1, get<2>(info), get<3>(info));
        user_array[user - 1].push_back(store_info);
    }

    return user_array;
}

vector<tuple<int, int, int>> *Data::format_user_data(int idx)
{
    switch (idx) {
        case 1:
            return get_user_data(array_1, ARRAY_1_SIZE);
        case 2:
            return get_user_data(array_2, ARRAY_2_SIZE);
        case 3:
            return get_user_data(array_3, ARRAY_3_SIZE);
        case 4:
            return get_user_data(array_4, ARRAY_4_SIZE);
        default:
            throw "Invalid index for array";
    }
}
