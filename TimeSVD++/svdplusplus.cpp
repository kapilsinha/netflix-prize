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

#define ARRAY_1_SIZE 94362233
#define ARRAY_2_SIZE 1965045
#define ARRAY_3_SIZE 1964391
#define ARRAY_4_SIZE 1374739
#define ARRAY_5_SIZE 2749898
#define NUM_USERS 458293

#define MAX_EPOCHS 200
#define EPS 0.001 // 0.0001

#define NUM_BINS 30
#define NUM_DAYS 2243

#define SCALE 0.9
#define BETA 0.4

#define sign(n) (n==0? 0 : (n<0?-1:1))

using namespace std;

// Let mu be the overall mean rating
const double mu = 3.6;
double G_alpha = 0.00001;        //gamma for alpha
const double L_alpha = 0.0004;   //learning rate for alpha
const double L_uv = 0.015;       //learning rate for U & V

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
    int userId, itemId, rating, timeval;
    // THIS IS NOT STOCHASTIC GRADIENT DESCENT ?????

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
        // tmpSum stores array of errors for each k?
        vector <double> tmpSum(K, 0);

        // populating sumMW (Line 90)
        // WHAT IS SUMMW ?????????? - KARTHIK
        // MAKE SURE k < K IS FINE - KARTHIK
        for (int k = 0; k < K; k++) {
            double sumy = 0;
            for (int i = 0; i < num_ratings; ++i) {
                int itemI = get<0>(ratings_info[userId][i]);
                sumy += y[itemI - 1][k];
            }
            sumMW[userId - 1][k] = sumy;
        }

        // Loop over all movies rated by this userId (Line 98)
        for (int i = 0; i < num_ratings; i++) {
            // DOUBLE CHECK THE INDICES - KARTHIK
            itemId = get<0>(ratings_info[userId][i]);
            rating = get<2>(ratings_info[userId][i]);
            timeval = get<1>(ratings_info[userId][i]);
            double predict = predictRating(userId, itemId, timeval);
            double error = rating - predict;
            // Subtract 1 because of indexing
            // userId -= 1;
            // itemId -= 1;
            // Update biases using gradients (Line 103)
            a[userId - 1] += eta * (error - reg * a[userId - 1]);
            b[itemId - 1] += eta * (error - reg * b[itemId - 1]);

            Bi_Bin[itemId - 1][calc_bin(timeval)] += eta * (error - reg * Bi_Bin[itemId - 1][calc_bin(timeval)]);
            Alpha_u[userId - 1] += G_alpha * (error * calc_dev_u(userId - 1,timeval)  - L_alpha * Alpha_u[userId - 1]);
            B_ut[userId - 1][timeval] += eta * (error - reg * B_ut[userId - 1][timeval]);


            // Update U and V using gradients (Line 106)
            for (int k = 0; k < K; k++) {
                auto uf = U[userId - 1][k];
                auto mf = V[itemId - 1][k];
                // AGAIN THE MAGICAL 0.015 COMING OUT OF ALADDIN'S ASS
                U[userId - 1][k] += eta * (error * mf - L_uv * uf);
                V[itemId - 1][k] += eta * (error * (uf + sqrtNum * sumMW[userId - 1][k]) - L_uv * mf);
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
                double tmpMW = y[itemId - 1][k];
                // WHY THE FUCK IS THERE A 0.015 HERE ?????????? - KARTHIK
                y[itemId - 1][k] += eta * (tmpSum[k] - L_uv * tmpMW);
                sumMW[userId - 1][k] += y[itemId - 1][k] - tmpMW;
            }
        }
    }

    // NO FUCKING CLUE WHAT THIS IS EITHER - KARTHIK (Line 123)
    for (userId = 1; userId <= M; userId++) {
        int num_ratings = ratings_info[userId].size();
        double sqrtNum = 0;
        if (num_ratings > 1) sqrtNum = 1 / sqrt(num_ratings);
        // LITERALLY NO IDEA WHAT ALL OF THIS IS ????????????
        for (int k = 0; k < K; k++) {
            double sumy = 0;
            for (int i = 0; i < num_ratings; i++) {
                int itemI = get<0>(ratings_info[userId][i]);
                sumy += y[itemI - 1][k];
            }
            sumMW[userId - 1][k] = sumy;
        }

    }
    progressBar.done();
    // THESE GUYS UPDATE LEARNING RATE HERE IDK IF WE WANNA DO THAT
    return;
}

double SVDPlusPlus::calc_dev_u(int user, int t)
{
    if(Dev[user].count(t)!=0) {
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
        vector<tuple<int, int, int>> *test_data, double reg, double *a, double *b)
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
    if (reg != 0) {
        double U_norm = 0.0;
        double V_norm = 0.0;
        double a_norm = 0.0;
        double b_norm = 0.0;
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                U_norm += U[row][col] * U[row][col];
            }
        }
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < K; col++) {
                V_norm += V[row][col] * V[row][col];
            }
        }
        for (int i = 0; i < M; i++) {
            a_norm += a[i] * a[i];
        }
        for (int j = 0; j < N; j++) {
            b_norm += b[j] * b[j];
        }
        err += 0.5 * reg * U_norm;
        err += 0.5 * reg * V_norm;
        err += 0.5 * reg * a_norm;
        err += 0.5 * reg * b_norm;
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
void SVDPlusPlus::train_model(int M, int N, int K, double eta,
        double reg, vector<tuple<int, int, int>> *ratings_info,
        vector<tuple<int, int, int>> *validation_ratings_info,
        vector<tuple<int, int, int>> *probe_ratings_info,
        double eps, int max_epochs) {
    cout << "Training model..." << endl;
    this->M = M;
    this->N = N;
    this->K = K;
    this->ratings_info = ratings_info;
    // Weird way to declare 2-D array but it allocates one contiguous block
    // stackoverflow.com/questions/29375797/copy-2d-array-using-memcpy/29375830

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

        for (int j = 0; j < ratings_info[i].size(); j++) {
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

    double delta;
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        cout << "Epoch: " << epoch << endl;
        is_trained = true;
        double before_E_in = get_err(U, V, ratings_info, reg, a, b);

        // Train the model
        Train(eta, reg);

        eta *= SCALE; // Scale learning rate by 0.9 after each epoch

        // Check early stopping conditions
        double E_in = get_err(U, V, ratings_info, reg, a, b);
        double E_val = get_err(U, V, validation_ratings_info, reg, a, b);
        double E_probe = get_err(U, V, probe_ratings_info, reg, a, b);
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
        dot_product += (U[i - 1][m] + sumMW[i - 1][m] * sq) * V[j - 1][m];
    }

    double rating = a[i - 1] + b[j - 1] + mu + dot_product + \
        Bi_Bin[j - 1][calc_bin(t)] + Alpha_u[i - 1]*calc_dev_u(i - 1,t) + B_ut[i - 1][t];

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

/**
 * Shouldn't be used ultimately (call methods in another file).
 * This is for testing purposes.
 */
/*
int main(void)
{
    // Arbitrarily chose set 3 and 5 as train and test sets
    // Test 2 (legitimate test)
    int train_set = 2;
    int Y_train_size = ARRAY_2_SIZE;

    int test_set = 3;
    int Y_test_size = ARRAY_3_SIZE;

    tuple<int, int, int> *Y_train = new tuple<int, int, int> [Y_train_size];
    tuple<int, int, int> *Y_test = new tuple<int, int, int> [Y_test_size];

    int M = 458293; // Number of users
    int N = 17770; // Number of movies
    int K = 20; // Number of factors

    Data data;
    tuple<int, int, int, int> *Y_train_original = data.getArray(train_set);
    tuple<int, int, int, int> *Y_test_original = data.getArray(test_set);

    for (int i = 0; i < Y_train_size; i++) {
        tuple<int, int, int, int> x = Y_train_original[i];
        Y_train[i] = make_tuple(get<0>(x), get<1>(x), get<3>(x));
    }
    for (int i = 0; i < Y_test_size; i++) {
        tuple<int, int, int, int> x = Y_test_original[i];
        Y_test[i] = make_tuple(get<0>(x), get<1>(x), get<3>(x));
    }

    double reg = 0.1;
    double eta = 0.01;

    SVDPlusPlus matfac;
    matfac.train_model(M, N, K, eta, reg, Y_train, Y_train_size, EPS, MAX_EPOCHS);

    // Get the errors
    double train_error = matfac.get_err(matfac.getU(), matfac.getV(),
                                        Y_train, Y_train_size, reg);
    double test_error = matfac.get_err(matfac.getU(), matfac.getV(),
                                       Y_test, Y_test_size, reg);

    // Add some more tests
    // (perhaps print out some values of U and V, errors, etc.)
    cout << "Some tests" << endl;
    cout << "Train error: " << train_error << endl;
    cout << "Test error: " << test_error << endl;
    cout << "\n" << endl;

    cout << "Training set predictions" << endl;
    for (int m = 0; m < 10; m++) {
        int i = get<0>(Y_train[m]);
        int j = get<1>(Y_train[m]);
        int Yij = get<2>(Y_train[m]);
        cout << "Y[" << i << "][" << j << "] = " << Yij << endl;
        cout << "Predicted value: " << matfac.predictRating(i, j) << endl;
    }

    cout << "\n" << endl;
    cout << "Test set predictions" << endl;
    for (int m = 0; m < 10; m++) {
        int i = get<0>(Y_test[m]);
        int j = get<1>(Y_test[m]);
        int Yij = get<2>(Y_test[m]);
        cout << "Y[" << i << "][" << j << "] = " << Yij << endl;
        cout << "Predicted value: " << matfac.predictRating(i, j) << endl;
    }
    return 0;
}

int main2(void)
{
    // Test 1 (very fast and simple)
    int M = 5;
    int N = 9;
    int K = 3;
    int Y_train_size = 30;
    int Y_test_size = 15;
    tuple<int, int, int> Y_train[30] =
    {make_tuple(1,1,3), make_tuple(1,2,4), make_tuple(1,3,5),
     make_tuple(1,7,1), make_tuple(1,8,2), make_tuple(1,9,2),
     make_tuple(2,4,2), make_tuple(2,5,3), make_tuple(2,6,4),
     make_tuple(2,7,4), make_tuple(2,8,1), make_tuple(2,9,5),
     make_tuple(3,1,1), make_tuple(3,2,1), make_tuple(3,3,4),
     make_tuple(3,4,2), make_tuple(3,5,2), make_tuple(3,6,4),
     make_tuple(4,1,1), make_tuple(4,2,1), make_tuple(4,3,4),
     make_tuple(4,7,3), make_tuple(4,8,1), make_tuple(4,9,5),
     make_tuple(5,4,1), make_tuple(5,5,2), make_tuple(5,6,2),
     make_tuple(5,7,3), make_tuple(5,8,5), make_tuple(5,9,3)};

    tuple<int, int, int> Y_test[15] =
    {make_tuple(1,4,3), make_tuple(1,5,2), make_tuple(1,6,1),
     make_tuple(2,1,4), make_tuple(2,2,2), make_tuple(2,3,2),
     make_tuple(3,7,3), make_tuple(3,8,1), make_tuple(3,9,5),
     make_tuple(4,4,2), make_tuple(4,5,2), make_tuple(4,6,4),
     make_tuple(5,1,1), make_tuple(5,2,4), make_tuple(5,3,4)};

    double reg = 0.1;
    double eta = 0.01;

    SVDPlusPlus matfac;
    matfac.train_model(M, N, K, eta, reg, Y_train, Y_train_size, EPS, MAX_EPOCHS);

    // Get the errors
    double train_error = matfac.get_err(matfac.getU(), matfac.getV(),
                                        Y_train, Y_train_size, reg);
    double test_error = matfac.get_err(matfac.getU(), matfac.getV(),
                                       Y_test, Y_test_size, reg);

    // Add some more tests
    // (perhaps print out some values of U and V, errors, etc.)
    cout << "Some tests" << endl;
    cout << "Train error: " << train_error << endl;
    cout << "Test error: " << test_error << endl;
    cout << "\n" << endl;

    cout << "Training set predictions" << endl;
    for (int m = 0; m < 10; m++) {
        int i = get<0>(Y_train[m]);
        int j = get<1>(Y_train[m]);
        int Yij = get<2>(Y_train[m]);
        cout << "Y[" << i << "][" << j << "] = " << Yij << endl;
        cout << "Predicted value: " << matfac.predictRating(i, j) << endl;
    }

    cout << "\n" << endl;
    cout << "Test set predictions" << endl;
    for (int m = 0; m < 10; m++) {
        int i = get<0>(Y_test[m]);
        int j = get<1>(Y_test[m]);
        int Yij = get<2>(Y_test[m]);
        cout << "Y[" << i << "][" << j << "] = " << Yij << endl;
        cout << "Predicted value: " << matfac.predictRating(i, j) << endl;
    }
    return 0;
}
*/
