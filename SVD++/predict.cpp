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
#define K 100 // Number of factors

using namespace std;

Data data;

/* Run the model. */
SVDPlusPlus* run_model(void) {
    // Set train and test set
    int train_set = 1; // Training set
    int val_set = 2; // Validation set
    int test_set = 4; // Probe set

    // Initialization
    vector<tuple<int, int, int>> *train_ratings_info = new vector<tuple<int, int, int>> [M];
    vector<tuple<int, int, int>> *val_ratings_info = new vector<tuple<int, int, int>> [M];
    vector<tuple<int, int, int>> *test_ratings_info = new vector<tuple<int, int, int>> [M];

    train_ratings_info = data.format_user_data(train_set);
    val_ratings_info = data.format_user_data(val_set);
    test_ratings_info = data.format_user_data(test_set);

    SVDPlusPlus *matfac = new SVDPlusPlus(M, N, K, train_ratings_info);
    matfac->train_model(val_ratings_info, test_ratings_info);

    // Get the errors
    double train_error = matfac->get_err(matfac->getU(), matfac->getV(),
                         train_ratings_info, matfac->getA(), matfac->getB());
    double val_error = matfac->get_err(matfac->getU(), matfac->getV(),
                       val_ratings_info, matfac->getA(), matfac->getB());
    double test_error = matfac->get_err(matfac->getU(), matfac->getV(),
                        test_ratings_info,  matfac->getA(), matfac->getB());

    cout << "Train error: " << train_error << endl;
    cout << "Validation error: " << val_error << endl;
    cout << "Test/probe error: " << test_error << endl;

    return matfac;
}

void write_preds(SVDPlusPlus *model) {
    string filename("SVD++_preds_" + to_string(K) + "_factors.txt");
    // If you want this to be descriptive, you have to return a string containing
    // the corresponding parameters from svdplusplus
    ofstream file(filename);
    if (file.is_open()) {
        // tuple<int, int, int> *qual = data.getQual();
        // Predict on the probe set
        tuple<int, int, int> *qual = data.getQual();
        for (int point = 0; point < ARRAY_5_SIZE; point++) {
            int i = get<0>(qual[point]);
            int j = get<1>(qual[point]);
            file << model->predictRating(i, j) << "\n";
        }
        file.close();
    }
    else {
        cout << "Unable to open file" << endl;
    }
}

void write_probe_preds(SVDPlusPlus *model) {
    string filename("SVD++_probe_preds_" + to_string(K) + "_factors.txt");
    // If you want this to be descriptive, you have to return a string containing
    // the corresponding parameters from svdplusplus
    ofstream file (filename);
    if (file.is_open()) {
        // tuple<int, int, int> *qual = data.getQual();
        // Predict on the probe set
        tuple<int, int, int, int> *probe = data.getArray(4);
        for (int point = 0; point < ARRAY_4_SIZE; point++) {
            // Some jank ass indexing shit if you wanna fix it be my guest
            int i = get<0>(probe[point]) - 1;
            int j = get<1>(probe[point]) - 1;
            file << model->predictRating(i, j) << "\n";
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
    cout << "Writing probe predictions..." << endl;
    write_probe_preds(model);
    return 0;
}
