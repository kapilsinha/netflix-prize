/**
 * @file matrix_factorization.hpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Performs matrix factorization on our movie rating data
 */

#include <string>
#include <tuple>
#include <array> // thin wrapper over bracket syntax (int [])
#include "readfile.hpp"

using namespace std;

class MatrixFactorization
{
private:
    int M;
    int N;
    int K;
    tuple<int, int, int> *Y;
    double **U;
    double **V;
    bool is_trained = false; // whether model has been trained or not
                             // i.e. if U and V contain meaningful values
    double *grad_U(double *Ui, int Yij, double *Vj, double reg, double eta);
    double *grad_V(double *Vj, int Yij, double *Ui, double reg, double eta);
public:
    MatrixFactorization(tuple<int, int, int> * Y); // Constructor
    ~MatrixFactorization(); // Destructor
    double get_err(double **U, double **V,
                   tuple<int, int, int> * Y, double reg = 0.0);
    // Trains model to generate U and V
    void train_model(int M, int N, int K, double eta, double reg,
            double eps = 0.0001, int max_epochs = 300);
    tuple<int, int, int> *getY(); // Returns Y
    double **getU(); // Returns U
    // double *getUi(int row); // Returns U[row]
    double **getV(); // Returns V
    // double *getVj(int row); // Returns V[row]
};
