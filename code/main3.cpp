#include <iostream>
#include <fstream> // for opening files
#include <vector>
#include <limits>    // for defining infinity
#include <cmath>     // for power function
#include <algorithm> // for std::max and std::min
#include <sstream>
#include <chrono>
using namespace std;
using namespace std::chrono;

// read data
void read_CSV(const string &filename, vector<double> &X, vector<int> &Y, size_t &n_features)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    string line;
    getline(file, line); // discard first line, which contains variable names

    stringstream ss(line);
    string cell;
    vector<string> fields;
    while (getline(ss, cell, ','))
        fields.push_back(cell);
    n_features = fields.size() - 1; // note that the last column is label

    while (getline(file, line))
    {

        stringstream ss(line);
        string cell;

        vector<string> fields;
        while (getline(ss, cell, ','))
            fields.push_back(cell);

        if (fields.size() < 2) // must have at least two features
            continue;

        for (size_t i = 0; i + 1 < fields.size(); ++i)
        {
            X.push_back(stod(fields[i]));
        }

        int label = stoi(fields.back());
        if (label != 1)
            label = -1;
        Y.push_back(label);
    }
}

// find min_val, max_val
void find_min_max(const vector<double> &X, vector<double> &min_val,
                  vector<double> &max_val, const size_t n_samples, const size_t n_features)
{
    for (size_t j = 0; j < n_features; ++j)
    {
        min_val[j] = X[j];
        max_val[j] = X[j];

        for (size_t i = 0; i < n_samples; ++i)
        {
            min_val[j] = min(min_val[j], X[i * n_features + j]);
            max_val[j] = max(max_val[j], X[i * n_features + j]);
        }
    }
}

// scale features
void scale_features(vector<double> &X,
                    const vector<double> &min_val, const vector<double> &max_val,
                    const size_t n_samples, const size_t n_features)
{
    for (size_t j = 0; j < n_features; ++j)
    {
        double range = max_val[j] - min_val[j];
        if (range < 1e-12)
            range = 1.0;

        for (size_t i = 0; i < n_samples; ++i)
        {
            X[i * n_features + j] = (X[i * n_features + j] - min_val[j]) / range;
        }
    }
}

// define kernel function
double kernel(const double *v1, const double *v2, const size_t n_features)
{
    // Gaussian kernel
    const double gamma = 0.00125;
    double res = 0.0;
    for (size_t i = 0; i < n_features; ++i)
    {
        res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    res = exp(-gamma * res);

    return res;
}

// calculate i_high
size_t calc_i_high(const vector<double> &alpha, const vector<int> &y, const vector<double> &f, const double C)
{
    const double eps = 1e-12;
    size_t ans = alpha.size();
    double min_val = numeric_limits<double>::infinity();

    for (size_t i = 0; i < alpha.size(); ++i)
    {
        bool in_I_high = ((y[i] == 1 && alpha[i] < C - eps) || (y[i] == -1 && alpha[i] > 0.0 + eps));
        if (in_I_high && f[i] < min_val)
        {
            min_val = f[i];
            ans = i;
        }
    }
    return ans;
}

// calculate i_low
size_t calc_i_low(const vector<double> &alpha, const vector<int> &y, const vector<double> &f, const double C)
{
    const double eps = 1e-12;
    size_t ans = alpha.size();
    double max_val = -numeric_limits<double>::infinity();

    for (size_t i = 0; i < alpha.size(); ++i)
    {
        bool in_I_low = ((y[i] == 1 && alpha[i] > 0.0 + eps) || (y[i] == -1 && alpha[i] < C - eps));
        if (in_I_low && f[i] > max_val)
        {
            max_val = f[i];
            ans = i;
        }
    }
    return ans;
}

// calculate lower bound, U, and upper bound, V, for alpha_i_low
bool calculate_U_V(const int s, const double alpha_i_high, const double alpha_i_low, const double C,
                   double &U, double &V)
{
    if (s == -1)
    {
        U = std::max(0.0, alpha_i_low - alpha_i_high);
        V = std::min(C, C + alpha_i_low - alpha_i_high);
    }
    else
    {
        U = std::max(0.0, alpha_i_low + alpha_i_high - C);
        V = std::min(C, alpha_i_low + alpha_i_high);
    }
    return U <= V + 1e-12;
}

// train data using SMO algorithm
void SMO_train(const vector<double> &x_train, const vector<int> &y_train, const size_t n_features,
               vector<double> &alpha, double &b, double C = 10.0)
{
    const double eps = 1e-12;
    size_t n = y_train.size();
    alpha.assign(n, 0.0);
    vector<double> f(n, 0.0);

    // initialize f = -y
    for (size_t i = 0; i < n; ++i)
        f[i] = -static_cast<double>(y_train[i]);

    // calclate train kernel matrix
    // vector<double> K_train(n * n, 0.0);
    // for (size_t i = 0; i < n; ++i)
    // {
    //     for (size_t j = i; j < n; ++j)
    //     {
    //         K_train[i * n + j] = kernel(x_train.data() + i * n_features, x_train.data() + j * n_features, n_features);
    //         if (j != i)
    //             K_train[j * n + i] = K_train[i * n + j];
    //     }
    // }

    size_t i_high;
    size_t i_low;
    double b_high;
    double b_low;

    size_t i_high_prev = n;          // only for updating K_i_high
    size_t i_low_prev = n;           // only for updating K_i_low
    vector<double> K_i_high(n, 0.0); // the row contain K(i_high, j) for j = 0,..n-1
    vector<double> K_i_low(n, 0.0);  // the row contain K(i_low, j) for j = 0,..n-1

    const double tau = 1e-5;
    int num_iter = 1;
    const int max_iter = 100000;

    while (true)
    {
        // update i_high, i_low, b_high, b_low
        i_high = calc_i_high(alpha, y_train, f, C);
        i_low = calc_i_low(alpha, y_train, f, C);
        if (i_high >= n || i_low >= n)
        {
            cerr << "i_high or i_low not found; iteration stops" << endl;
            break;
        }
        b_high = f[i_high];
        b_low = f[i_low];

        if (b_low <= b_high + 2.0 * tau)
            break;

        if (i_high != i_high_prev)
        {
            // update i_high_prev and K_i_high
            // in the first iteration, i_high_prev is set to n, so i_high != i_high_prev, there must be an update
            i_high_prev = i_high;
            for (size_t j = 0; j < n; ++j)
                K_i_high[j] = kernel(x_train.data() + i_high * n_features, x_train.data() + j * n_features, n_features);
        }

        if (i_low != i_low_prev)
        {
            // update i_low_prev and K_i_low
            // in the first iteration, i_low_prev is set to n, so i_low != i_low_prev, there must be an update
            i_low_prev = i_low;
            for (size_t j = 0; j < n; ++j)
                K_i_low[j] = kernel(x_train.data() + i_low * n_features, x_train.data() + j * n_features, n_features);
        }

        // update alpha
        int s = y_train[i_high] * y_train[i_low];
        // double K11 = K_train[i_high * n + i_high];
        // double K22 = K_train[i_low * n + i_low];
        // double K12 = K_train[i_high * n + i_low];
        double K11 = K_i_high[i_high];
        double K22 = K_i_low[i_low];
        double K12 = K_i_high[i_low]; // or K_i_low[i_high]

        double eta = K11 + K22 - 2.0 * K12;

        double U, V;
        if (!calculate_U_V(s, alpha[i_high], alpha[i_low], C, U, V))
        {
            cerr << "warning: infeasible U and V; iteration stops" << endl;
            break;
        }

        double next_alpha_i_low = alpha[i_low];
        if (eta <= eps)
        {
            cerr << "warning: non positive eta; iteration stops" << endl;
            break;
        }

        next_alpha_i_low = alpha[i_low] + y_train[i_low] * (b_high - b_low) / eta;
        // clip to [U, V]
        if (next_alpha_i_low > V)
            next_alpha_i_low = V;
        if (next_alpha_i_low < U)
            next_alpha_i_low = U;

        double next_alpha_i_high = alpha[i_high] + s * (alpha[i_low] - next_alpha_i_low);

        // update f
        double delta_alpha_i_high = next_alpha_i_high - alpha[i_high];
        double delta_alpha_i_low = next_alpha_i_low - alpha[i_low];
        for (size_t i = 0; i < n; ++i)
        {
            // f[i] += delta_alpha_i_high * y_train[i_high] * K_train[i_high * n + i] + delta_alpha_i_low * y_train[i_low] * K_train[i_low * n + i];
            f[i] += delta_alpha_i_high * y_train[i_high] * K_i_high[i] + delta_alpha_i_low * y_train[i_low] * K_i_low[i];
        }

        // updtae alpha
        alpha[i_high] = next_alpha_i_high;
        alpha[i_low] = next_alpha_i_low;

        ++num_iter;
        // avoid infinite loop
        if (num_iter > max_iter)
        {
            cerr << "Too many iterations; stopping" << endl;
            break;
        }
    }

    cout << "number of iterations: " << num_iter << endl;
    b = (b_high + b_low) / 2;
    cout << "b = " << std::fixed << std::setprecision(15) << b << endl;
    cout << "(b_high - b_low)/2*1e10 = " << std::fixed << std::setprecision(15) << (b_high - b_low) / 2 * 1e10 << endl;
}

// Given alpha and tol, return indices of support vectors (alpha > tol)
vector<int> get_SV_indices(const vector<double> &alpha, double tol = 1e-8)
{
    vector<int> idx;
    for (size_t i = 0; i < alpha.size(); ++i)
        if (alpha[i] > tol)
            idx.push_back((int)i);
    return idx;
}

int main()
{
    // string dataset = "debug"; // C = 1, gamma = 0.125
    // string dataset = "mnist"; // C = 10, gamma = 0.00125
    // string dataset = "mnist2"; // C = 10, gamma = 0.00125
    string dataset = "mnist3"; // C = 10, gamma = 0.00125
    // string dataset = "banknote"; // C = 1, gamma = 0.125
    // string dataset = "sample";
    string train_file = dataset + "_train_data.csv";

    // read train dataset
    vector<double> x_train;
    vector<int> y_train;
    size_t n_features;
    read_CSV(train_file, x_train, y_train, n_features);

    size_t n = x_train.size() / n_features;
    cout << "n = " << n << endl;
    cout << "n_features = " << n_features << endl;
    if (n == 0)
    {
        cerr << "Error: No data read from file." << endl;
        return 1;
    }

    vector<double> min_val(n_features, 0.0);
    vector<double> max_val(n_features, 0.0);

    // starting timer
    auto start = high_resolution_clock::now();

    // scale train data
    find_min_max(x_train, min_val, max_val, n, n_features);
    scale_features(x_train, min_val, max_val, n, n_features);

    // hyperparameters
    double C = 10.0;
    vector<double> alpha(n, 0.0);
    double b;

    // train data uisng SMO algorithm
    SMO_train(x_train, y_train, n_features, alpha, b, C);

    // read test dataset
    string test_file = dataset + "_test_data.csv";
    vector<double> x_test;
    vector<int> y_test;
    read_CSV(test_file, x_test, y_test, n_features);
    size_t m = x_test.size() / n_features;
    scale_features(x_test, min_val, max_val, m, n_features);

    // calclate train-test kernel matrix
    // vector<double> K_test_train(m * n, 0.0);
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         K_test_train[i * n + j] = kernel(x_test.data() + i * n_features, x_train.data() + j * n_features, n_features);
    //     }
    // }

    double tol = 1e-8;
    vector<int> sv_idx = get_SV_indices(alpha, tol);
    int sv_count = (int)sv_idx.size();
    cout << "Final SV count = " << sv_count << "\n";

    // end timer
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start);

    // make prediction
    vector<int> y_pred(m);
    int correct = 0;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     double curr = -b; // note that the prediction function is sign(wx -b)
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         curr += alpha[j] * y_train[j] * kernel(x_test.data() + i * n_features, x_train.data() + j * n_features, n_features);
    //     }
    //     y_pred[i] = curr > 0 ? 1 : -1;
    //     if (y_pred[i] == y_test[i])
    //         correct++;
    // }

    for (size_t i = 0; i < m; ++i)
    {
        double curr = -b; // note that the prediction function is sign(wx -b)
        for (size_t k = 0; k < sv_count; ++k)
        {
            size_t j = sv_idx[k];
            curr += alpha[j] * y_train[j] * kernel(x_test.data() + i * n_features, x_train.data() + j * n_features, n_features);
        }
        y_pred[i] = curr > 0 ? 1 : -1;
        if (y_pred[i] == y_test[i])
            correct++;
    }

    double accuracy = double(correct) / double(m);
    cout << "Test accuracy = " << std::fixed << std::setprecision(15) << accuracy << " (" << correct << "/" << m << ")\n";

    // end timer
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2 - end1);
    auto duration3 = duration_cast<milliseconds>(end2 - start);

    cout << "Training time: " << duration1.count() << " ms" << endl;
    cout << "Prediction time: " << duration2.count() << " ms" << endl;
    cout << "Total Runtime: " << duration3.count() << " ms" << endl;

    return 0;
}