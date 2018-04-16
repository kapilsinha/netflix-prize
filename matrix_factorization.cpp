/**
 * @file matrix_factorization.cpp
 * @author Kapil Sinha
 * @date 04/06/18
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
// #include <chrono> // std::chrono::system_clock
// #include <iterator> // std::begin, std::end
// #include <array>

#include "matrix_factorization.hpp"
// #include "readfile.hpp"

using namespace std;

/**
 * @brief Constructs a MatrixFactorization instance, which contains the sparse
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

// Actually I don't see the value in storing Y anymore...
MatrixFactorization::MatrixFactorization(tuple<int, int, int> *Y)
{
    this->Y = Y;
}

/**
 * @brief Destructs a MatrixFactorization instance.
 */
MatrixFactorization::~MatrixFactorization()
{
    delete[] Y;
    Y = nullptr;
    if (is_trained) {
        // Strange way of deleting variables but it fits the way I
        // initialized it
        delete[] U[0];
        delete[] U;
        delete[] V[0];
        delete[] V;
    }
    else {
        delete[] U;
        delete[] V;
    }
    U = nullptr;
    V = nullptr;
}

/**
 * @brief Computes gradient of regularized loss function with respect to Ui
 * multiplied by eta
 *
 * @param
 * Ui : ith row of U
 * Yij : training point
 * Vj : jth column of V^T (jth row of V)
 * reg : regularization parameter lambda
 * eta : learning rate
 *
 * @return gradient * eta
 */
double *MatrixFactorization::grad_U(double *Ui, int Yij,
                        double *Vj, double reg, double eta)
{
    // Based off of CS 155 solutions (check it)
    double * gradient = new double [K];
    for (int m = 0; m < K; m++) {
        gradient[m] = (1 - reg * eta) * Ui[m]
                   + eta * Vj[m] * (Yij - Ui[m] * Vj[m]);
    }
    return gradient;
}

/**
 * @brief Computes gradient of regularized loss function with respect to Vj
 * multiplied by eta
 *
 * @param
 * Vj : jth column of V^T (jth row of V)
 * Yij : training point
 * Ui : ith row of U
 * reg : regularization parameter lambda
 * eta : learning rate
 *
 * @return gradient * eta
 */
double *MatrixFactorization::grad_V(double *Vj, int Yij,
                        double *Ui, double reg, double eta)
{
    // Based off of CS 155 solutions (check it)
    double * gradient = new double [K];
    for (int m = 0; m < K; m++) {
        gradient[m] = (1 - reg * eta) * Vj[m]
                   + eta * Ui[m] * (Yij - Ui[m] * Vj[m]);
    }
    return gradient;
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
double MatrixFactorization::get_err(double **U,
        double **V, tuple<int, int, int> *Y, double reg /* = 0.0 */)
{
    // Based off of CS 155 solutions (check it)
    double err = 0.0;
    // Hack to calculate length (from geeksforgeeks 8) ) - check that it works
    // maybe can use sizeof ?
    int Y_size = *(&Y + 1) - Y;
    for (int m = 0; m < Y_size; m++) {
        int i = get<0>(Y[m]);
        int j = get<1>(Y[m]);
        int Yij = get<2>(Y[m]);

        double dot_product = 0;
        for (int n = 0; n < K; n++) {
            dot_product += U[i - 1][n] * V[j - 1][n];
        }
        err += 0.5 * (Yij - dot_product) * (Yij - dot_product);

        if (reg != 0) {
            double U_frobenius_squared_norm = 0;
            double V_frobenius_squared_norm = 0;
            for (int row = 0; row < M; row++) {
                for (int col = 0; col < K; col++) {
                    U_frobenius_squared_norm += U[row][col] * U[row][col];
                }
            }
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < K; col++) {
                    V_frobenius_squared_norm += V[row][col] * V[row][col];
                }
            }
            err += 0.5 * reg * U_frobenius_squared_norm;
            err += 0.5 * reg * V_frobenius_squared_norm;
        }
    }
    return err / Y_size;
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
 */

void MatrixFactorization::train_model(int M, int N, int K, double eta,
        double reg, double eps /* = 0.0001 */, int max_epochs /* = 300 */) {
    // Based off of CS 155 solutions
    this->M = M;
    this->N = N;
    this->K = K;
    // Weird way to declare 2-D array but it allocates one contiguous block
    // and allows for indexing normally e.g. U[0][1] is row 0, col 1
    // Note that we store V^T, not V (i.e. N x K matrix, not K x N)
    // stackoverflow.com/questions/29375797/copy-2d-array-using-memcpy/29375830
    U = new double*[M];
    U[0] = new double[M * K];
    for (int i = 1; i < M; i++) {
        U[i] = U[i - 1] + K;
    }
    V = new double*[N];
    V[0] = new double[N * K];
    for (int j = 1; j < N; j++) {
        V[j] = V[j - 1] + K;
    }

    std::random_device rd;  // Will be used to obtain a seed for random engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dist(-0.5, 0.5); // Uniform distribution

    // Initialize the values of U to be uniform between -0.5 and 0.5
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            U[i][j] = dist(gen);
        }
    }
    // Initialize the values of V to be uniform between -0.5 and 0.5
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            V[i][j] = dist(gen);
        }
    }

    // Hack to calculate length (from geeksforgeeks 8) ) - check that it works
    int Y_size = *(&Y + 1) - Y;
    double delta;

    // Creates list of indices so we can shuffle them later
    // http://en.cppreference.com/w/cpp/algorithm/iota
    std::list<int> indices(Y_size);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        cout << "Epoch: " << epoch << endl;
        double before_E_in = get_err(U, V, Y, reg);
        std::vector<std::list<int>::iterator> shuffled_indices(indices.size());
        std::iota(shuffled_indices.begin(),
                  shuffled_indices.end(), indices.begin());
        std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);

        // auto requires C++11
        for (auto ind: shuffled_indices) {
            int i = get<0>(Y[*ind]);
            int j = get<1>(Y[*ind]);
            int Yij = get<2>(Y[*ind]);

            // Update the row of U using the gradient
            // Note: the gradient function actually returns U[i - 1] - gradient
            // so we simply set U[i - 1] to this value (instead of subtracting
            // the gradient)
            double *gradient_U = grad_U(U[i - 1], Yij, V[j - 1], reg, eta);
            double *gradient_V = grad_V(V[j - 1], Yij, U[i - 1], reg, eta);
            for (int index = 0; index < K; index++) {
                U[i - 1][index] = gradient_U[index];
                V[j - 1][index] = gradient_V[index];
            }
            // Freeing the dynamically allocated gradient
            delete[] gradient_U;
            delete[] gradient_V;
        }

        // Check early stopping conditions
        double E_in = get_err(U, V, Y, reg);
        if (epoch == 0) {
            delta = before_E_in - E_in;
        }
        else if (before_E_in - E_in < eps * delta) {
            break;
        }
    }
    is_trained = true;
    return;
}

/**
 * @brief Returns Y array (sparse matrix represented as 3-tuples)
 * @return Y
 */
tuple<int, int, int> * MatrixFactorization::getY()
{
    return Y;
}

/**
 * @brief Returns U array (sparse matrix represented as 3-tuples)
 * @return U
 */
double **MatrixFactorization::getU()
{
    return U;
}

/**
 * @brief Returns V array (sparse matrix represented as 3-tuples)
 * @return V
 */
double **MatrixFactorization::getV()
{
    return V;
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
int main(void)
{
    // Set these variables - you can probably set the Y_train by calling
    // the Data class and wrangling with it to remove the date and set the
    // M, N, and K to the appropriate sizes (we can vary K)
    //
    // Arbitrarily chose set 3 and 5 as train and test sets
    int ARRAY_3_SIZE = 1964391;
    int ARRAY_5_SIZE = 2749898;
    tuple<int, int, int> *Y_train = new tuple<int, int, int> [ARRAY_5_SIZE];
    tuple<int, int, int> *Y_test = new tuple<int, int, int> [ARRAY_3_SIZE];
    int M = 458293;
    int N = 17770;
    int K = 1000;
    double reg = 0.0;
    double eta = 0.03;

    Data data;
    tuple<int, int, int, int> *Y_train_original = data.getArray(5);
    tuple<int, int, int, int> *Y_test_original = data.getArray(3);
    for (int i = 0; i < ARRAY_5_SIZE; i++) {
        tuple<int, int, int, int> x = Y_train_original[i];
        Y_train[i] = make_tuple(get<0>(x), get<1>(x), get<3>(x));
    }
    for (int i = 0; i < ARRAY_3_SIZE; i++) {
        tuple<int, int, int, int> x = Y_test_original[i];
        Y_test[i] = make_tuple(get<0>(x), get<1>(x), get<3>(x));
    }
    MatrixFactorization matfac(Y_train);
    matfac.train_model(M, N, K, eta, reg);
    double train_error = matfac.get_err(matfac.getU(), matfac.getV(), Y_train);
    double test_error = matfac.get_err(matfac.getU(), matfac.getV(), Y_test);

    // Add some more tests
    // (perhaps print out some values of U and V, errors, etc.)
    cout << "Some checks" << endl;
    cout << "Train error: " << train_error << endl;
    cout << "Test error: " << test_error << endl;

    return 0;
}
