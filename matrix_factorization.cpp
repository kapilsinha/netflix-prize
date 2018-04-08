/**
 * @file matrix_factorization.cpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Performs matrix factorization on our movie rating data
 */

#include <iostream>
#include <tuple>
#include "matrix_factorization.hpp"
#include "readfile.hpp"

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
 * but I am keeping this class general so that bridge will need to be made
 * later.
 */
MatrixFactorization::MatrixFactorization(tuple<int, int, int> * Y)
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
double MatrixFactorization::grad_U(tuple<int, int, int> Ui, int Yij,
                        tuple<int, int, int> Vj, double reg, double eta)
{
    // Based off of CS 155 solutions (check it)
    double elem1 = (1 - reg * eta) * get<0>(Ui)
                   + eta * get<0>(Vj) * (Yij - get<0>(Ui) * get<0>(Vj))
    double elem2 = (1 - reg * eta) * get<1>(Ui)
                   + eta * get<1>(Vj) * (Yij - get<1>(Ui) * get<1>(Vj))
    double elem3 = (1 - reg * eta) * get<2>(Ui)
                   + eta * get<2>(Vj) * (Yij - get<2>(Ui) * get<2>(Vj))
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
double MatrixFactorization::grad_V(tuple<int, int, int> Vj, int Yij,
                        tuple<int, int, int> Ui, double reg, double eta)
{
    // Based off of CS 155 solutions (check it)
    double elem1 = (1 - reg * eta) * get<0>(Vj)
                   + eta * get<0>(Ui) * (Yij - get<0>(Ui) * get<0>(Vj))
    double elem2 = (1 - reg * eta) * get<1>(Vj)
                   + eta * get<1>(Ui) * (Yij - get<1>(Ui) * get<1>(Vj))
    double elem3 = (1 - reg * eta) * get<2>(Vj)
                   + eta * get<2>(Ui) * (Yij - get<2>(Ui) * get<2>(Vj))
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
double MatrixFactorization::get_err(tuple<int, int, int> * U,
        tuple<int, int, int> * V, tuple<int, int, int> * Y, double reg = 0.0)
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
    return 0.0;
}
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
        double reg, tuple<int, int, int> * Y, double eps = 0.0001,
        int max_epochs = 300)
{
    // Based off of CS 155 solutions (check it)
    // TODO
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
tuple<int, int, int> * MatrixFactorization::getU()
{
    return U;
}

/**
 * @brief Returns V array (sparse matrix represented as 3-tuples)
 * @return V
 */
tuple<int, int, int> * MatrixFactorization::getV()
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
    // TODO
    // Set these variables - you can probably set the Y_train by calling
    // the Data class and wrangling with it to remove the date and set the
    // M, N, and K to the appropriate sizes (we can vary K)
    tuple<int, int, int> * Y_train;
    tuple<int, int, int> * Y_test;
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
    print_tuple (data.getU()[0]);
    cout << "V element 1: ";
    print_tuple (data.getV()[0]);
    cout << "Y element 1: ";
    print_tuple (data.getY()[0]);

    return 0;
}
