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
#include "ProgressBar.hpp"

using namespace std;

class SVDPlusPlus
{
private:
    int M;
    int N;
    int K;
    double **U;
    double **V;
    double *a;
    double *b;
    bool is_trained = false; // whether model has been trained or not
                             // i.e. if U and V contain meaningful values
    double *grad_U(double *Ui, int Yij, double *Vj, double reg, double eta, double ai, double bj);
    double *grad_V(double *Vj, int Yij, double *Ui, double reg, double eta, double ai, double bj);
    double *grad_A(double *Ui, int Yij, double *Vj, double reg, double eta, double ai, double bj);
    double *grad_B(double *Ui, int Yij, double *Vj, double reg, double eta, double ai, double bj);
    double *grad_y(double *Ui, int Yij,double *Vj, double reg, double eta, double ai, double bj);
public:
    SVDPlusPlus(); // Constructor
    ~SVDPlusPlus(); // Destructor
    double get_err(double **U, double **V, tuple<int, int, int> * Y,
            int Y_length, double reg, double *a, double *b);
    // Trains model to generate U and V
    void train_model(int M, int N, int K, double eta, double reg,
            tuple<int, int, int> *Y, int Y_length,
            double eps, int max_epochs);
    double predictRating(int i, int j);
    double **getU(); // Returns U
    // double *getUi(int row); // Returns U[row]
    double **getV(); // Returns V
    // double *getVj(int row); // Returns V[row]
    double *getA();
    double *getB();
};
