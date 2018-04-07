/**
 * @file matrix_factorization.hpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Performs matrix factorization on our movie rating data
 */

#include <string>
#include <tuple>
#include "readfile.hpp"

using namespace std;

class MatrixFactorization
{
private:
    tuple<int, int, int> * Y;
    tuple<int, int, int> * U;
    tuple<int, int, int> * V;
    double grad_U(tuple<int, int, int> Ui, int Yij,
            tuple<int, int, int> Vj, double reg, double eta);
    double grad_V(tuple<int, int, int> Vj, int Yij,
            tuple<int, int, int> Ui, double reg, double eta);
public:
    MatrixFactorization(tuple<int, int, int> * Y); // Constructor
    ~MatrixFactorization(); // Destructor
    double get_err(tuple<int, int, int> * U, tuple<int, int, int> * V,
                   tuple<int, int, int> * Y, double reg = 0.0);
    void train_model(int M, int N, int K, double eta, double reg,
            tuple<int, int, int> * Y, double eps = 0.0001,
            int max_epochs = 300); // Trains model to generate U and V
    tuple<int, int, int> * getY(); // Returns Y
    tuple<int, int, int> * getU(); // Returns U
    tuple<int, int, int> * getV(); // Returns V
};
