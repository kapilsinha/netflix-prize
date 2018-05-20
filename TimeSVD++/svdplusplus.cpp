/**
 * @file svdplusplus.cpp
 * @author Karthik Karnik
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

#define NUM_BINS 30
#define NUM_DAYS 2243

#define SCALE 0.9
#define BETA 0.4
#define MAX_EPOCHS 40
#define EPS 0.001

#define MU 3.513
#define L_ALPHA 0.0004
#define L_UV 0.015
#define L 0.005
#define DECAY 0.9

#define sign(n) (n==0? 0 : (n<0?-1:1))

using namespace std;

// Let mu be the overall mean rating
double G_alpha = 0.00001;        // gamma for alpha
double G = 0.007;                // general gamma

/**
 * @brief Constructs a SVDPlusPlus instance, which contains the sparse
 * input data (arranged in 3-tuples), and the generated matrix factors U and V
 * such that rating Y[i][j] is approximated by (UV)[i][j]
 *
 * @param
 * Y : input matrix
 *
 * NOTE: The Data class outputs an array of 4-tuples including date, but this
 * class takes an input of 3-tuples ignoring date. To make them work together,
 * you need to remove the date element from the elements. I could do that here
 * but I am keeping this class general so the bridge will need to be made
 * later.
 */

// Actually I don't see the value in storing Y anymore... (I dont see the value in life anymore)
SVDPlusPlus::SVDPlusPlus(int M, int N, int K, vector<tuple<int, int, int>> *ratings_info)
{
	this->M = M;
	this->N = N;
	this->K = K;
	this->ratings_info = ratings_info;
	// Initialize all arrays
    U = new double *[M];
    U[0] = new double [M * K];
    for (int i = 1; i < M; i++) {
        U[i] = U[i - 1] + K;
    }
    V = new double *[N];
    V[0] = new double [N * K];
    for (int j = 1; j < N; j++) {
        V[j] = V[j - 1] + K;
    }
    y = new double *[N];
    y[0] = new double [N * K];
    for (int i = 1; i < N; i++) {
        y[i] = y[i - 1] + K;
    }
    sumMW = new double *[M];
    sumMW[0] = new double [M * K];
    for (int i = 1; i < M; i++) {
        sumMW[i] = sumMW[i - 1] + K;
    }
    // Bias stuff
    a = new double [M];
    b = new double [N];

    // Initialize Dev array for deviation
    Dev = new map<int,double> [M];

    // Initialize Tu
    Tu = new double[M];
    for (int i = 0; i < M; i++) {
        double temp = 0;
        int num_ratings = ratings_info[i].size();

        if (num_ratings == 0){
            Tu[i] = 0;
        }

        else {
            for (int j = 0; j < num_ratings; j++) {
                temp += get<1>(ratings_info[i][j]);
            }

            Tu[i] = temp / num_ratings;
        }
    }

    //Initialize Alpha_u
    Alpha_u = new double[M];
    for (int i = 0; i < M; i++){
        Alpha_u[i] = 0.0;
    }

    // Initialize Bi_Bin
    Bi_Bin = new double* [N];

    for (int i = 0; i < N; i++) {
        Bi_Bin[i] = new double[NUM_BINS];
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < NUM_BINS; j++){
            Bi_Bin[i][j] = 0.0;
        }
    }

    // Initialize B_ut
    B_ut = new map<int,double> [M];

    for (int i = 0; i < M; i++) {
        map<int,double> temp;

        for (unsigned int j = 0; j < ratings_info[i].size(); j++) {
            int date = get<1>(ratings_info[i][j]);
            if (temp.count(date) == 0) {
                temp[date] = 0.0000001;
            }

            else continue;
        }

        B_ut[i] = temp;
    }


    std::random_device rd;  // Will be used to obtain a seed for random engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dist(-0.5, 0.5); // Uniform distribution

    // Initialize all values to be uniform between -0.5 and 0.5
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            U[i][j] = dist(gen);
            sumMW[i][j] = dist(gen);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            V[i][j] = dist(gen);
            y[i][j] = 0;
        }
    }
    for (int i = 0; i < M; i++) {
        a[i] = dist(gen);
    }
    for (int i = 0; i < N; i++) {
        b[i] = dist(gen);
    }
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
 * eta : learning rate
 * ratings_info : array of vector of tuples where i^th vector corresponds to ratings
 *              of i^th user
 *
 * @return gradient * eta
 */
void SVDPlusPlus::Train()
{
    ProgressBar progressBar(M, 70);
    int userId, itemId, rating, timeval;

    for (userId = 0; userId < M; userId++) {
        ++progressBar;
        progressBar.display();
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

        // Loop over all movies rated by this userId (Line 98)
        for (int i = 0; i < num_ratings; i++) {

            itemId = get<0>(ratings_info[userId][i]);
            rating = get<2>(ratings_info[userId][i]);
            timeval = get<1>(ratings_info[userId][i]);
            double predict = predictRating(userId, itemId, timeval);
            double error = rating - predict;
       
            a[userId] += G * (error - L * a[userId]);
            b[itemId] += G * (error - L * b[itemId]);

            Bi_Bin[itemId][calc_bin(timeval)] += G * (error - L * Bi_Bin[itemId][calc_bin(timeval)]);
            Alpha_u[userId] += G_alpha * (error * calc_dev_u(userId,timeval)  - L_ALPHA * Alpha_u[userId]);
            B_ut[userId][timeval] += G * (error - L * B_ut[userId][timeval]);


            // Update U and V using gradients (Line 106)
            for (int k = 0; k < K; k++) {
                auto uf = U[userId][k];
                auto mf = V[itemId][k];
                // AGAIN THE MAGICAL 0.015 COMING OUT OF ALADDIN'S ASS
                U[userId][k] += G * (error * mf - L_UV * uf);
                V[itemId][k] += G * (error * (uf + sqrtNum * sumMW[userId][k]) - L_UV * mf);
                tmpSum[k] += error * sqrtNum * mf;
            }
        }

        // Update sumMW and y (Line 114)
        // MAYBE PUT THIS IN THE LOOP ABOVE???
        // COMPLETELY UNCLEAR ON THE LOGIC HERE OR WTF IS GOING SOMEBODY PLS EXPLAIN - KARTHIK
        // ????????????????????????????????????????????????????????????????????????????????
        for (int j = 0; j < num_ratings; ++j) {
            itemId = get<0>(ratings_info[userId][j]);
            for (int k = 0; k < K; k++) {
                double tmpMW = y[itemId][k];
                // WHY THE FUCK IS THERE A 0.015 HERE ?????????? - KARTHIK
                y[itemId][k] += G * (tmpSum[k] - L_UV * tmpMW);
                sumMW[userId][k] += y[itemId][k] - tmpMW;
            }
        }
    }

    // NO FUCKING CLUE WHAT THIS IS EITHER - KARTHIK (Line 123)
    for (userId = 0; userId < M; userId++) {
        int num_ratings = ratings_info[userId].size();
        // LITERALLY NO IDEA WHAT ALL OF THIS IS ????????????
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
    G *= DECAY;
    G_alpha *= DECAY; 
    return;
}

double SVDPlusPlus::calc_dev_u(int user, int t)
{
    if(Dev[user].count(t) != 0) {
        return Dev[user][t];
    }

    double temp = sign(t - Tu[user]) * pow(double(abs(t - Tu[user])), BETA);
    Dev[user][t] = temp;
    return temp;
}

int SVDPlusPlus::calc_bin(int t)
{
    int size_of_bin = NUM_DAYS/NUM_BINS + 1;
    return t / size_of_bin;
}

/**
 * @brief Computes mean squared-error of predictions made by
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
    // SOMEONE CHECK THIS FUNCTION AT SOME POINT - KARTHIK
    double err = 0.0;
    int num = 0;
    // Loop over users
    for (int userId = 0; userId < M; userId++) {
        int num_ratings = test_data[userId].size();
        // Loop over training points
        for (int itemI = 0; itemI < num_ratings; itemI++) {
            int itemId = get<0>(test_data[userId][itemI]);
            int rating = get<2>(test_data[userId][itemI]);
            int timeval = get<1>(test_data[userId][itemI]);
            double predict = predictRating(userId, itemId, timeval);
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
 * Y : input matrix
 * eps : fraction where if MSE between epochs is less than
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
double SVDPlusPlus::predictRating(int i, int j, int t)
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

    double rating = a[i] + b[j] + MU + dot_product + \
        Bi_Bin[j][calc_bin(t)] + Alpha_u[i] * calc_dev_u(i,t) + B_ut[i][t];

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