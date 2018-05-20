/**
 * @file matrix_factorization.hpp
 * @author Kapil Sinha
 * @date 04/06/18
 *
 * @brief Performs matrix factorization on our movie rating data
 */
#include <map>
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
    bool is_trained = false; // whether model has been trained or not
                             // i.e. if U and V contain meaningful values
public:
    SVDPlusPlus();
    ~SVDPlusPlus();
    double **U;
    double **V;
    double **y;
    double **sumMW;
    double *a;
    double *b;

    map<int,double> *Dev;
    double *Tu;           //variable for mean time of user
    double *Alpha_u;
    double **Bi_Bin;
    map<int,double> *B_ut;

    vector<tuple<int, int, int>> *ratings_info;

    double get_err(double **U, double **V,
        vector<tuple<int, int, int>> *test_data, double reg, double *a, double *b);
    // Trains model to generate U and V
    void train_model(int M, int N, int K, double eta, double reg,
            vector<tuple<int, int, int>> *ratings_info,
            vector<tuple<int, int, int>> *validation_ratings_info,
            vector<tuple<int, int, int>> *probe_ratings_info,
            double eps, int max_epochs);
    double predictRating(int i, int j, int t);
    void Train(double reg, double eta);

    //double predictRating(int i, int j);
    double **getU(); // Returns U
    // double *getUi(int row); // Returns U[row]
    double **getV(); // Returns V
    // double *getVj(int row); // Returns V[row]
    double *getA();
    double *getB();

    double calc_dev_u(int user, int t);
    int calc_bin(int t);


};