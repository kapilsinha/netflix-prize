/**
 * @file predict.cpp
 * @author King Nair
 * @date 5/1/18
 *
 * @brief Actually runs the shit
 */
#include "predict.hpp"
#include <string>

#define ARRAY_1_SIZE 94362233 // Training data
#define ARRAY_2_SIZE 1965045 // Validation data
#define ARRAY_3_SIZE 1964391 // Hidden data
#define ARRAY_4_SIZE 1374739 // Probe into test set
#define ARRAY_5_SIZE 2749898 // Qual data

#define M 458293 // Number of users
#define N 17770 // Number of movies
#define K 10 // Number of factors

#define REG 0.00 // Regularization
#define ETA 0.01 // Learning rate
#define MAX_EPOCHS 100
#define EPS 0.001 // 0.0001

using namespace std;

/* Run the model. */
SVDPlusPlus *run_model(void) {
    // Set train and test set
    int train_set = 1; // Training set
    int test_set = 2; // Validation set
    Data data;
    /*
    int Y_train_size = ARRAY_1_SIZE;
    int Y_test_size = ARRAY_4_SIZE;

    // Initialization
    tuple<int, int, int> *Y_train = new tuple<int, int, int> [Y_train_size];
    tuple<int, int, int> *Y_test = new tuple<int, int, int> [Y_test_size];

    tuple<int, int, int, int> *Y_train_original = data.getArray(train_set);
    tuple<int, int, int, int> *Y_test_original = data.getArray(test_set);

    // Get rid of the dates
    for (int i = 0; i < Y_train_size; i++) {
        tuple<int, int, int, int> x = Y_train_original[i];
        Y_train[i] = make_tuple(get<0>(x), get<1>(x), get<3>(x));
    }
    for (int i = 0; i < Y_test_size; i++) {
        tuple<int, int, int, int> x = Y_test_original[i];
        Y_test[i] = make_tuple(get<0>(x), get<1>(x), get<3>(x));
    }
    */

    // Initialization
    vector<tuple<int, int, int>> *train_ratings_info = new vector<tuple<int, int, int>> [M];
    vector<tuple<int, int, int>> *test_ratings_info = new vector<tuple<int, int, int>> [M];

    train_ratings_info = data.format_user_data(train_set);
    test_ratings_info = data.format_user_data(test_set);


    SVDPlusPlus *matfac = new SVDPlusPlus();
    matfac->train_model(M, N, K, ETA, REG, train_ratings_info, test_ratings_info, EPS, MAX_EPOCHS);

    // Get the errors
    double train_error = matfac->get_err(matfac->getU(), matfac->getV(),
                                        train_ratings_info, REG, matfac->getA(), matfac->getB());
    double test_error = matfac->get_err(matfac->getU(), matfac->getV(),
                                       test_ratings_info, REG,  matfac->getA(), matfac->getB());

    cout << "Train error: " << train_error << endl;
    cout << "Test error: " << test_error << endl;

    return matfac;
}

void write_preds(SVDPlusPlus *model) {
    string filename ("Time_SVD_predictions_");
    filename += to_string(K) + "factors_" + to_string(REG) + "reg_"
             + to_string(ETA) + "eta_" + to_string(MAX_EPOCHS)
             + "epochs_" + to_string(EPS) + "eps.txt";
    ofstream file (filename);
    if (file.is_open()) {
        Data data;
        tuple<int, int, int> *qual = data.getQual();
        for (int point = 0; point < ARRAY_5_SIZE; point++) {
            int i = get<0>(qual[point]);
            int j = get<1>(qual[point]);
            int t = get<2>(qual[point]);
            file << model->predictRating(i, j, t) << "\n";
        }
        file.close();
    }
    else {
        cout << "Unable to open file" << endl;
    }
}

int main(void)
{
    cout << "Running the model..." << endl;
    SVDPlusPlus *model = run_model();
    cout << "Writing the predictions..." << endl;
    write_preds(model);
    return 0;
}
