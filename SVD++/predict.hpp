/**
 * @file predict.cpp
 * @date 5/1/18
 *
 * @brief Runs SVD++ and makes predictions
 */

#include <iostream>
#include <fstream>
#include <algorithm> // std::shuffle
#include <tuple>
#include <list> // couldn't figure out how not to use this for shuffling
#include <vector> // couldn't figure out how not to use this for shuffling
#include <random> // std::random_device, std::mt19937,
                  // std::uniform_real_distribution
#include <numeric> // std::iota
#include <math.h> // sqrt

using namespace std;

#include "svdplusplus.hpp"

SVDPlusPlus *run_model(void);
void write_preds(SVDPlusPlus *model);
