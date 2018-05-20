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
    array_1 = new tuple<int, int, int, int> [ARRAY_1_SIZE];
    array_2 = new tuple<int, int, int, int> [ARRAY_2_SIZE];
    array_3 = new tuple<int, int, int, int> [ARRAY_3_SIZE];
    array_4 = new tuple<int, int, int, int> [ARRAY_4_SIZE];
    array_5 = new tuple<int, int, int> [ARRAY_5_SIZE];

    int array_1_index = 0;
    int array_2_index = 0;
    int array_3_index = 0;
    int array_4_index = 0;
    int array_5_index = 0;

    ifstream infile("mu/all.dta");
    ifstream idxfile("mu/all.idx");

    while (!infile.eof() & !idxfile.eof() ) {
        int user, movie, date, rating;
        int idx;
        infile >> user >> movie >> date >> rating;
        idxfile >> idx;
        // Populate the arrays
        switch(idx) {
            case 1:
                array_1[array_1_index] = make_tuple(user, movie, date, rating);
                array_1_index++;
                break;
            case 2:
                array_2[array_2_index] = make_tuple(user, movie, date, rating);
                array_2_index++;
                break;
            case 3:
                array_3[array_3_index] = make_tuple(user, movie, date, rating);
                array_3_index++;
                break;
            case 4:
                array_4[array_4_index] = make_tuple(user, movie, date, rating);
                array_4_index++;
                break;
            case 5:
                array_5[array_5_index] = make_tuple(user, movie, date);
                array_5_index++;
                break;
            default:
                // Ideally show what the index is but this error is more of a
                // formality and I don't know how to do that
                throw "Unexpected index in all.idx";
        }
    }
}

/**
 * @brief Destructs a Data instance.
 */
Data::~Data()
{
    delete[] array_1;
    array_1 = NULL;
    delete[] array_2;
    array_2 = NULL;
    delete[] array_3;
    array_3 = NULL;
    delete[] array_4;
    array_4 = NULL;
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
tuple<int, int, int, int> *Data::getArray(int idx)
{
    switch (idx) {
        case 1:
            return array_1;
        case 2:
            return array_2;
        case 3:
            return array_3;
        case 4:
            return array_4;
        default:
            throw "Invalid index for array";
    }
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
        tuple<int, int, int> store_info = make_tuple(get<1>(info), get<2>(info), get<3>(info));
        user_array[user].push_back(store_info);
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

/**
 * Shouldn't be used ultimately (call methods in another file).
 * This is for testing purposes.
 */
/*
int main(void)
{
    Data data;
    cout << "Some checks" << endl;
    cout << "Array 1 element 0: ";
    print_tuple (data.getArray(1)[0]);
    cout << "Array 2 element 0: ";
    print_tuple (data.getArray(2)[0]);
    cout << "Array 3 element 0: ";
    print_tuple (data.getArray(3)[0]);
    cout << "Array 4 element 0: ";
    print_tuple (data.getArray(4)[0]);
    cout << "Array 5 element 0: ";
    print_tuple (data.getArray(5)[0]);

    cout << "Array 1 last element: ";
    print_tuple (data.getArray(1)[ARRAY_1_SIZE - 1]);
    cout << "Array 2 last element: ";
    print_tuple (data.getArray(2)[ARRAY_2_SIZE - 1]);
    cout << "Array 3 last element: ";
    print_tuple (data.getArray(3)[ARRAY_3_SIZE - 1]);
    cout << "Array 4 last element: ";
    print_tuple (data.getArray(4)[ARRAY_4_SIZE - 1]);
    cout << "Array 5 last element: ";
    print_tuple (data.getArray(5)[ARR5_SIZE - 1]);

    cout << "Array 1 garbage: ";
    print_tuple (data.getArray(1)[ARRAY_1_SIZE]);
    cout << "Array 2 garbage: ";
    print_tuple (data.getArray(2)[ARRAY_2_SIZE]);
    cout << "Array 3 garbage: ";
    print_tuple (data.getArray(3)[ARRAY_3_SIZE]);
    cout << "Array 4 garbage: ";
    print_tuple (data.getArray(4)[ARRAY_4_SIZE]);
    cout << "Array 5 garbage: ";
    print_tuple (data.getArray(5)[ARRAY_5_SIZE]);

    return 0;
}
*/