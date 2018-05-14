/**
 * @file matrix_factorization.cpp
 * @author Kapil Sinha (Here just in case anyone forgets who made this)...ye
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

#include "matrix_factorization.hpp"

#define ARRAY_1_SIZE 94362233
#define ARRAY_2_SIZE 1965045
#define ARRAY_3_SIZE 1964391
#define ARRAY_4_SIZE 1374739
#define ARRAY_5_SIZE 2749898

#define MAX_EPOCHS 200
#define EPS 0.001 // 0.0001

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

// Actually I don't see the value in storing Y anymore... (I dont see the value in life anymore)
MatrixFactorization::MatrixFactorization()
{
}

/**
 * @brief Destructs a MatrixFactorization instance.
 */
MatrixFactorization::~MatrixFactorization()
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
                        double *Vj, double reg, double eta, double ai, double bj)
{
    double *gradient = new double [K];
    double dot_product = 0.0;
    // Compute the dot product
    for (int i = 0; i < K; i++) {
        dot_product += Ui[i] * Vj[i];
    }
    // Compute the gradient
    // MULTIPLIED BY 2
    for (int m = 0; m < K; m++) {
        gradient[m] = (1 - reg * eta) * Ui[m]
                   + eta * Vj[m] * 2 * (Yij - dot_product - ai - bj);
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
                        double *Ui, double reg, double eta, double ai, double bj)
{
    double *gradient = new double [K];
    double dot_product = 0.0;
    // Compute the dot product
    for (int i = 0; i < K; i++) {
        dot_product += Ui[i] * Vj[i];
    }
    // Compute the gradient
    // MULTIPLIED BY 2
    for (int m = 0; m < K; m++) {
        gradient[m] = (1 - reg * eta) * Vj[m]
                   + eta * Ui[m] * 2 * (Yij - dot_product - ai - bj);
    }
    return gradient;
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
double *MatrixFactorization::grad_A(double *Ui, int Yij,
                        double *Vj, double reg, double eta, double ai, double bj)
{
    double *gradient = new double [M];
    double dot_product = 0.0;
    // Compute the dot product
    for (int i = 0; i < K; i++) {
        dot_product += Ui[i] * Vj[i];
    }
    // Compute the gradient
    for (int m = 0; m < M; m++) {
        gradient[m] = (1 - reg * eta) * ai
                   + eta * 2 * (Yij - dot_product - ai - bj);
    }
    return gradient;
}

/**
 * @brief Computes gradient of regularized loss function with respect to B
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
double *MatrixFactorization::grad_B(double *Ui, int Yij,
                        double *Vj, double reg, double eta, double ai, double bj)
{
    double *gradient = new double [N];
    double dot_product = 0.0;
    // Compute the dot product
    for (int i = 0; i < K; i++) {
        dot_product += Ui[i] * Vj[i];
    }
    // Compute the gradient
    for (int m = 0; m < N; m++) {
        gradient[m] = (1 - reg * eta) * bj
                   + eta * 2 * (Yij - dot_product - ai - bj);
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
double MatrixFactorization::get_err(double **U, double **V,
        tuple<int, int, int> *Y, int Y_length, double reg, double *a, double *b)
{
    // Based off of CS 155 solutions (check it)
    double err = 0.0;
    for (int m = 0; m < Y_length; m++) {
        int i = get<0>(Y[m]);
        int j = get<1>(Y[m]);
        int Yij = get<2>(Y[m]);

        double dot_product = 0.0;
        for (int n = 0; n < K; n++) {
            dot_product += U[i - 1][n] * V[j - 1][n];
        }
        // cout << "ai " << a[i] << "  bj " << b[j] << endl;
        err += 0.5 * (Yij - dot_product - a[i] - b[j]) * (Yij - dot_product - a[i] - b[j]);
    }

    if (reg != 0) {
        double U_frobenius_squared_norm = 0.0;
        double V_frobenius_squared_norm = 0.0;
        double a_frobenius_squared_norm = 0.0;
        double b_frobenius_squared_norm = 0.0;
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
        for (int i = 0; i < M; i++) {
            a_frobenius_squared_norm += a[i] * a[i];
        }
        for (int j = 0; j < N; j++) {
            b_frobenius_squared_norm += b[j] * b[j];
        }
        err += 0.5 * reg * U_frobenius_squared_norm;
        err += 0.5 * reg * V_frobenius_squared_norm;
        err += 0.5 * reg * a_frobenius_squared_norm;
        err += 0.5 * reg * b_frobenius_squared_norm;
    }
    cout << "err " << err << endl;
    if (err < 0) {
        cout << "ABORT: error below zero" << endl;
    }
    return err / Y_length;
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
        double reg, tuple<int, int, int> *Y, int Y_length,
        double eps, int max_epochs) {
    cout << "Training model..." << endl;
    // Based off of CS 155 solutions
    this->M = M;
    this->N = N;
    this->K = K;
    // Weird way to declare 2-D array but it allocates one contiguous block
    // and allows for indexing normally e.g. U[0][1] is row 0, col 1
    // Note that we store V^T, not V (i.e. N x K matrix, not K x N)
    // stackoverflow.com/questions/29375797/copy-2d-array-using-memcpy/29375830
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

    //Bias stuff 
    a = new double [M];
    b = new double [N];
    // Initialize the values of a to be uniform between -0.5 and 0.5
    for (int i = 0; i < M; i++) {
          a[i] = dist(gen);
    }
      // Initialize the values of  b to be uniform between -0.5 and 0.5
    for (int i = 0; i < N; i++) {
        b[i] = dist(gen);
    }

    double delta;
    // Creates list of indices so we can shuffle them later
    // http://en.cppreference.com/w/cpp/algorithm/iota
    std::list<int> indices(Y_length);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        cout << "Epoch: " << epoch << endl;
        double before_E_in = get_err(U, V, Y, Y_length, reg, a, b);

        std::vector<std::list<int>::iterator> shuffled_indices(indices.size());
        std::iota(shuffled_indices.begin(),
                  shuffled_indices.end(), indices.begin());
        std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);

        //cout << "A" << endl; 
        // auto requires C++11
        for (auto ind: shuffled_indices) {
            int i = get<0>(Y[*ind]);
            int j = get<1>(Y[*ind]);
            int Yij = get<2>(Y[*ind]);

            // Update the row of U using the gradient
            // Note: the gradient function actually returns U[i - 1] - gradient
            // so we simply set U[i - 1] to this value (instead of subtracting
            // the gradient)
            double *gradient_U = grad_U(U[i - 1], Yij, V[j - 1], reg, eta, a[i - 1], b[j - 1]);
            // DON'T SCREW UP THE BELOW BY SWITCHING U AND V!!!
            double *gradient_V = grad_V(V[j - 1], Yij, U[i - 1], reg, eta, a[i - 1], b[j - 1]);

            // 
            double *gradient_A = grad_A(U[i - 1], Yij, V[j - 1], reg, eta, a[i - 1], b[j - 1]);
            double *gradient_B = grad_B(U[i - 1], Yij, V[j - 1], reg, eta, a[i - 1] , b[j - 1]);

            //cout << "B i: " << i << "    j: " << j <<  "   M: "<< M<<"   N:  "<< N << endl; 
            for (int index = 0; index < K; index++) {
                U[i - 1][index] = gradient_U[index];
                V[j - 1][index] = gradient_V[index];
            }

            for (int l = 0; l < M; l++) {
                a[l] = gradient_A[l];
            }

            for (int l = 0; l < N; l++) {
                b[l] = gradient_B[l];
            }

            // Freeing the dynamically allocated gradient
            delete[] gradient_U;
            delete[] gradient_V;
            delete[] gradient_A;
            delete[] gradient_B;
        }

        // Check early stopping conditions
        // Can be optimized because we are computing before_E_in twice at each loop
        // ^^ not sure what you mean...
        double E_in = get_err(U, V, Y, Y_length, reg, a, b);
        if (epoch == 0) {
            delta = before_E_in - E_in;
        }
        else if (before_E_in - E_in < eps * delta) {
            cout << "eps" << eps;
            cout<< "delta" << delta;
            cout << "Error: " << E_in;
            cout << ", Delta error: " << (before_E_in - E_in);
            cout << ", Threshold delta error: " << (eps * delta) << endl;
            break;
        }
        else {
            cout << "Error: " << E_in;
            cout << ", Delta error: " << (before_E_in - E_in);
            cout << ", Threshold delta error: " << (eps * delta) << endl;
        }
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
double MatrixFactorization::predictRating(int i, int j)
{
    if (!is_trained) {
        cout << "Model not trained yet!" << endl;
        return 0;
    }
    double rating = 0;
    for (int m = 0; m < K; m++) {
        rating += U[i - 1][m] * V[j - 1][m] + a[i - 1] + b[j - 1];
    }
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
 * @brief Returns A array 
 * @return A
 */
double *MatrixFactorization::getA()
{
    return a;
}

/**
 * @brief Returns B array 
 * @return B
 */
double *MatrixFactorization::getB()
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

    MatrixFactorization matfac;
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

    MatrixFactorization matfac;
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

