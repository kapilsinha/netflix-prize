/**
 * @file matrix_factorization.cpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Performs matrix factorization on our movie rating data
 */

#include <iostream>
#include <tuple>
#include <array>
#include "matrix_factorization.hpp"
#include "readfile.hpp"

using namespace std;

int

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
 * but I am keeping this class general so that bridge will need to be made
 * later.
 */
MatrixFactorization::MatrixFactorization(tuple<int, int, int> *Y)
{
    Y = this.Y;
}

/**
 * @brief Destructs a MatrixFactorization instance.
 */
MatrixFactorization::~MatrixFactorization()
{
    delete[] Y;
    Y = NULL;
    delete[] U;
    U = NULL;
    delete[] V;
    V = NULL;
}

/**
 * @brief Computes gradient of regularized loss function with respect to Ui
 * multiplied by eta
 *
 * @param
 * Ui : ith row of U
 * Yij : training point
 * Vj : jth column of V^T
 * reg : regularization parameter lambda
 * eta : learning rate
 *
 * @return gradient * eta
 */
array<double, K> MatrixFactorization::grad_U(array<int, 3> Ui, int Yij,
                        array<int, 3> Vj, double reg, double eta)
{
    // Based off of CS 155 solutions (check it)
    double elem1 = (1 - reg * eta) * (Ui[0])
                   + eta * (Vj[0]) * (Yij - (Ui[0]) * (Vj[0]))
    double elem2 = (1 - reg * eta) * (Ui[1])
                   + eta * (Vj[1]) * (Yij - (Ui[1]) * (Vj[1]))
    double elem3 = (1 - reg * eta) * (Ui[2])
                   + eta * (Vj[2]) * (Yij - (Ui[2]) * (Vj[2]))
    return make_tuple(elem1, elem2, elem3);
}

/**
 * @brief Computes gradient of regularized loss function with respect to Vj
 * multiplied by eta
 *
 * @param
 * Vj : jth column of V^T
 * Yij : training point
 * Ui : ith row of U
 * reg : regularization parameter lambda
 * eta : learning rate
 *
 * @return gradient * eta
 */
double MatrixFactorization::grad_V(array<int, 3> Vj, int Yij,
                        array<int, 3> Ui, double reg, double eta)
{
    // Based off of CS 155 solutions (check it)
    double elem1 = (1 - reg * eta) * (Vj[0])
                   + eta * (Ui[0]) * (Yij - (Ui[0]) * (Vj[0]))
    double elem2 = (1 - reg * eta) * (Vj[1])
                   + eta * (Ui[1]) * (Yij - (Ui[1]) * (Vj[1]))
    double elem3 = (1 - reg * eta) * (Vj[2])
                   + eta * (Ui[2]) * (Yij - (Ui[2]) * (Vj[2]))
    return make_tuple(elem1, elem2, elem3);
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
double MatrixFactorization::get_err(array<int, 3> *U,
        array<int, 3> *V, array<int, 3> *Y, double reg = 0.0)
{
    // Based off of CS 155 solutions (check it)
    // TODO
    double err = 0.0;
    // Hack to calculate length (from geeksforgeeks 8)) - check that it works
    int Y_size = *(&Y + 1) - Y;
    for (int m = 0; m < Y_size; m++) {
        int i = get<0>(Y[m]);
        int j = get<1>(Y[m]);
        int Yij = get<2>(Y[m]);
        // FILL IN THE REST
        // I think we always access columns of V (rows of V^T) so it may be
        // better to store V^T instead of V since accessing columns is expensive
    }
    return err;
}

/**
 * @param
 * M : number of rows in U (U is an M x K matrix)
 * N : number of rows in V (V is an N x K matrix)
 * K : number of columns in U and V
 * eta : learning rate
 * reg : regularization constant
 * Y : input matrix
 * eps : fraction where if regularized MSE between epochs is less than
 *       eps times the decrease in MSE after the first epoch, we stop training
 * max_epochs : maximum number of epochs for training
 */

void MatrixFactorization::train_model(int M, int N, int K, double eta,
        double reg, array<array<int, 3>, /* TODO: Enter length */> Y,
    double eps = 0.0001,
        int max_epochs = 300) {
    // Based off of CS 155 solutions
    double prev_err = 9999999;
    double prev_diff = 9999999;

    std::random_device rd;  // Will be used to obtain a seed for random engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-0.5, 0.5); // Uniform distribution

    // Define the U matrix with M rows and K columns
    array<array<double, K>, M> U;
    // Initialize the values of U to be uniform between -0.5 and 0.5
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            U[i][j] = dis(gen);
        }
    }
    // Define the V matrix with N rows and K columns
    array<array<double, K>, N> V;
    // Initialize the values of V to be uniform between -0.5 and 0.5
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            V[i][j] = dis(gen);
        }
    }

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // TODO: Find random permutation of Y
        for (int point = 0; point < Y.size(); point++) {
            array<int, 3> y = Y[point];
            int user = y[0];
            int movie = y[1];
            int Yij = y[3];

            // Find the row and column to be updated
            array<double, K> Ui = U[y[0]];
            array<double, K> Vj = V[y[1]];

            // Update the row of U using the gradient
            array<double, K> gradient_U = grad_U(Ui, Yij, Vj, reg, eta);
            for (int i = 0; i < gradient_U.size(); i++) {
                U[user][i] -= gradient_U[i]
            }

            // Update the row of V using the gradient
            array<double, K> gradient_V = grad_V(Ui, Yij, Vj, reg, eta);
            for (int i = 0; i < gradient_V.size(); i++) {
                V[movie][i] -= gradient_V[i]
            }

        }

        // Check early stopping conditions
        double curr_err = get_err(U, V, Y)
        if (prev_diff / (curr_err - prev_err) < eps) {
            break;
        }
        else {
            prev_diff = curr_err - prev_err;
            prev_err = curr_err;
        }

    }
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
tuple<int, int, int> *MatrixFactorization::getU()
{
    return U;
}

/**
 * @brief Returns V array (sparse matrix represented as 3-tuples)
 * @return V
 */
tuple<int, int, int> *MatrixFactorization::getV()
{
    return V;
}

// /**
//  * Used only for testing purposes (printing in main method)
//  */
// void print_tuple(tuple<int, int, int> tup) {
//     cout << "(" << get<0>(tup) << " " << get<1>(tup) << " "
//          << get<2>(tup) << " " << ")" << endl;
// }

/**
 * Shouldn't be used ultimately (call methods in another file).
 * This is for testing purposes.
 */
int main(void)
{
    // TODO
    // Set these variables - you can probably set the Y_train by calling
    // the Data class and wrangling with it to remove the date and set the
    // M, N, and K to the appropriate sizes (we can vary K)
    tuple<int, int, int> *Y_train;
    tuple<int, int, int> *Y_test;
    int M;
    int N;
    int K;
    double reg = 0.0;
    double eta = 0.03;

    MatrixFactorization matfac;
    matfac.train_model(M, N, K, eta, reg, Y_train);
    double error = matfac.get_err(matfac.U, matfac.V, Y_test);

    // Add some more tests
    // (perhaps print out some values of U and V, errors, etc.)
    cout << "Some checks" << endl;

    cout << "Error: " << error << endl;
    cout << "U element 1: ";
    // print_tuple (data.getU()[0]);
    cout << "V element 1: ";
    // print_tuple (data.getV()[0]);
    cout << "Y element 1: ";
    // print_tuple (data.getY()[0]);

    return 0;
}
