/**
 * @file svdplusplus.cpp
 * @date 05/14/18
 *
 * @brief Performs matrix factorization on our movie rating data
 */

#include <iostream>
#include <algorithm> // std::shuffle
#include <tuple>
#include <list> // couldn't figure out how not to use this for shuffling
#include <vector> // couldn't figure out how not to use this for shuffling
#include <random> // std::random_device, std::mt19937,
                  // std::uniform_real_distribution
#include <numeric> // std::iota

#include "svdplusplus.hpp"

#define ARRAY_1_SIZE 94362233
#define ARRAY_2_SIZE 1965045
#define ARRAY_3_SIZE 1964391
#define ARRAY_4_SIZE 1374739
#define ARRAY_5_SIZE 2749898
#define NUM_USERS 458293

#define MAX_EPOCHS 200
#define EPS 0.001 // 0.0001

using namespace std;

// Let mu be the overall mean rating
const double mu = 3.6;


/**
 * @brief Constructs a SVDPlusPlus instance, which contains the sparse
 * input data (arranged in 3-tuples), and the generated matrix factors U and V
 * such that rating Y[i][j] is approximated by (UV)[i][j]
 *
 * @param
 * Y : input matrix
 */

SVDPlusPlus::SVDPlusPlus()
{
}

/**
 * @brief Destructs a SVDPlusPlus instance.
 */
SVDPlusPlus::~SVDPlusPlus()
{
    if (is_trained) {
        // Strange way of deleting variables but it fits the way I
        // initialized it
        delete[] U[0];
        delete[] U;
        delete[] V[0];
        delete[] V;
    }
    U = nullptr;
    V = nullptr;
}

/**
 * @brief Computes and updates all of the gradients
 *
 * @param
 * i : user
 * j : movie
 * rating : rating
 * reg : regularization parameter lambda
 * eta : learning rate
 * ratings_info : array of vector of tuples where i^th vector corresponds to ratings
 *              of i^th user
 *
 * @return gradient * eta
 */
void SVDPlusPlus::Train(double eta, double reg)
{
    ProgressBar progressBar(M, 100);
    int userId, itemId, rating;

    // RANDOMIZING 
    std::random_device rd;  // Will be used to obtain a seed for random engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    std::list<int> indices(M);
    std::iota(indices.begin(), indices.end(), 1);

    std::vector<std::list<int>::iterator> shuffled_indices(indices.size());
    std::iota(shuffled_indices.begin(),
              shuffled_indices.end(), indices.begin());
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);

    // END RANDOMIZING
    for (auto ind: shuffled_indices) {
        ++progressBar;
        progressBar.display();
        userId = *ind;
        // Number of ratings for this user
        int num_ratings = ratings_info[userId].size();
        double sqrtNum = 0;
        if (num_ratings > 1) sqrtNum = 1 / sqrt(num_ratings);
        // tmpSum stores array of errors for each k
        vector <double> tmpSum(K, 0);

        // populating sumMW
        for (int k = 0; k < K; k++) {
            double sumy = 0;
            for (int i = 0; i < num_ratings; ++i) {
    U = nullptr;
    V = nullptr;
}

/**
 * @brief Computes and updates all of the gradients
 *
 * @param
 * i : user
 * j : movie
 * rating : rating
 * reg : regularization parameter lambda
 * eta : learning rate
 * ratings_info : array of vector of tuples where i^th vector corresponds to ratings
 *              of i^th user
 *
 * @return gradient * eta
 */
void SVDPlusPlus::Train()
{
    ProgressBar progressBar(M, 100);
    int userId, itemId, rating;

    // RANDOMIZING

    /*
    std::random_device rd;  // Will be used to obtain a seed for random engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    std::list<int> indices(M);
    std::iota(indices.begin(), indices.end(), 1);

    std::vector<std::list<int>::iterator> shuffled_indices(indices.size());
    std::iota(shuffled_indices.begin(),
              shuffled_indices.end(), indices.begin());
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);
    */


    // END RANDOMIZING
    // for (auto ind: shuffled_indices) {
    for (userId = 0; userId < NUM_USERS; userId++) {
        ++progressBar;
        progressBar.display();
        //userId = *ind;
        // Number of ratings for this user
        int num_ratings = ratings_info[userId].size();
        double sqrtNum = 0;
        if (num_ratings > 1) sqrtNum = 1 / sqrt(num_ratings);
        // tmpSum stores array of errors for each k?
        vector <double> tmpSum(K, 0);

        // populating sumMW
        for (int k = 0; k < K; k++) {
            double sumy = 0;
            for (int i = 0; i < num_ratings; ++i) {
                int itemI = get<0>(ratings_info[userId][i]);
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }

        // Loop over all movies rated by this userId
        for (int i = 0; i < num_ratings; i++) {
            itemId = get<0>(ratings_info[userId][i]);
            rating = get<2>(ratings_info[userId][i]);
            double predict = predictRating(userId, itemId);
            double error = rating - predict;
            // Subtract 1 because of indexing
            // userId -= 1;
            // itemId -= 1;
            // Update biases using gradients (Line 103)
            a[userId] += eta * (error - reg * a[userId]);
            b[itemId] += eta * (error - reg * b[itemId]);

            // Update U and V using gradients (Line 106)
            for (int k = 0; k < K; k++) {
                auto uf = U[userId][k];
                auto mf = V[itemId][k];
                // AGAIN THE MAGICAL 0.015 COMING OUT OF ALADDIN'S ASS
                U[userId][k] += eta * (error * mf - L_UV * uf);
                V[itemId][k] += eta * (error * (uf + sqrtNum * sumMW[userId][k]) - L_UV * mf);
                tmpSum[k] += error * sqrtNum * mf;
            }
        }

        // Update sumMW and y
        for (int j = 0; j < num_ratings; ++j) {
            itemId = get<0>(ratings_info[userId][j]);
            for (int k = 0; k < K; k++) {
                double tmpMW = y[itemId][k];
                y[itemId][k] += eta * (tmpSum[k] - L_UV * tmpMW);
                sumMW[userId][k] += y[itemId][k] - tmpMW;
            }
        }
    }

    for (userId = 0; userId < M; userId++) {
        int num_ratings = ratings_info[userId].size();
        //double sqrtNum = 0;
        //if (num_ratings > 1) sqrtNum = 1 / sqrt(num_ratings);

        for (int k = 0; k < K; k++) {
            double sumy = 0;
            for (int i = 0; i < num_ratings; i++) {
                int itemI = get<0>(ratings_info[userId][i]);
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }

    }
    progressBar.done();
    eta *= DECAY;
    return;
}



/**
 * @brief Computes mean regularized squared-error of predictions made by
 * estimating Y[i][j] as the dot product of U[i] and V[j]
 *
 * @param
 * Y : matrix of triples (i, j, Y_ij)
 *     where i is the index of a user,
 *           j is the index of a movie,
 *       and Y_ij is user i's rating of movie j and user/movie matrices U and V
 * U : user matrix (factor of Y)
 * V : movie matrix (factor of Y)
 *
 * @return error (MSE)
 */
 double SVDPlusPlus::get_err(double **U, double **V,
         vector<tuple<int, int, int>> *test_data, double *a, double *b)
 {
     double err = 0.0;
     int num = 0;
     // Loop over users
     for (int userId = 0; userId < M; userId++) {
         int num_ratings = test_data[userId].size();
         // Loop over training points
         for (int itemI = 0; itemI < num_ratings; itemI++) {
             int itemId = get<0>(test_data[userId][itemI]);
             int rating = get<2>(test_data[userId][itemI]);
             double predict = predictRating(userId, itemId);
             err += (predict - rating) * (predict - rating);
             num++;
         }
     }
     return sqrt(err / num);
 }


/**
 * @param
 * M : number of rows in U (U is an M x K matrix)
 * N : number of rows in V^T (V^T is an N x K matrix)
 * K : number of columns in U and V^T
 * eta : learning rate
 * reg : regularization constant
 * Y : input matrix
 * eps : fraction where if regularized MSE between epochs is less than
 *       eps times the decrease in MSE after the first epoch, we stop training
 * max_epochs : maximum number of epochs for training
 * ratings_info : array of vector of tuples where i^th vector corresponds to ratings
 *              of i^th user
 */
void SVDPlusPlus::train_model(vector<tuple<int, int, int>> *validation_ratings_info,
        vector<tuple<int, int, int>> *probe_ratings_info) {
    cout << "Training model..." << endl;
    this->M = M;
    this->N = N;
    this->K = K;

    double eps = EPS;
    double max_epochs = MAX_EPOCHS;
    double delta;
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        cout << "Epoch: " << epoch << endl;
        is_trained = true;
        double before_E_in = get_err(U, V, ratings_info, a, b);

        // Train the model
        Train();

        // Check early stopping conditions
        double E_in = get_err(U, V, ratings_info, a, b);
        double E_val = get_err(U, V, validation_ratings_info, a, b);
        double E_probe = get_err(U, V, probe_ratings_info, a, b);
        if (epoch == 0) {
            delta = before_E_in - E_in;
        }
        else if (before_E_in - E_in < eps * delta) {
            cout << "eps " << eps;
            cout<< ", delta " << delta;
            cout << ", Error: " << E_in;
            cout << ", Delta error: " << (before_E_in - E_in);
            cout << ", Threshold delta error: " << (eps * delta) << endl;
            break;
        }
        else {
            cout << "Error: " << E_in;
            cout << ", Delta error: " << (before_E_in - E_in);
            cout << ", Threshold delta error: " << (eps * delta) << endl;
        }
        cout << "Validation error: " << E_val << endl;
        cout << "Probe error: " << E_probe << endl;
    }
    is_trained = true;
    return;
}

/**
 * @brief Predicts rating given a user and movie.
 * Note that it subtracts one from the input indices to compute the
 * matrix outputs (since we start indexing at 0, not 1)
 * @return predicted rating
 */
double SVDPlusPlus::predictRating(int i, int j)
{
    if (!is_trained) {
        cout << "Model not trained yet!" << endl;
        return 0;
    }

    // Compute the predicted rating
    int num_ratings = ratings_info[i].size();
    double sq = 0;
    if (num_ratings > 1) {
        sq = 1. / sqrt(num_ratings);
    }

    double dot_product = 0;
    for (int m = 0; m < K; m++) {
        dot_product += (U[i][m] + sumMW[i][m] * sq) * V[j][m];
    }

    double rating = a[i] + b[j] + mu + dot_product;

    // Cap the ratings
    if (rating < 1) {
        rating = 1;
    }
    if (rating > 5) {
        rating = 5;
    }
    return rating;
}

/**
 * @brief Returns U array (sparse matrix represented as 3-tuples)
 * @return U
 */
double **SVDPlusPlus::getU()
{
    return U;
}

/**
 * @brief Returns V array (sparse matrix represented as 3-tuples)
 * @return V
 */
double **SVDPlusPlus::getV()
{
    return V;
}


/**
 * @brief Returns A array
 * @return A
 */
double *SVDPlusPlus::getA()
{
    return a;
}

/**
 * @brief Returns B array
 * @return B
 */
double *SVDPlusPlus::getB()
{
    return b;
}

/**
 * Used only for testing purposes (printing in main method)
 */
void print_tuple(tuple<int, int, int> tup) {
    cout << "(" << get<0>(tup) << " " << get<1>(tup) << " "
         << get<2>(tup) << " " << ")" << endl;
}
