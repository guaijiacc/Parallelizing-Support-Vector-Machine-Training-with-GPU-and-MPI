#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <iomanip>
using namespace std;

// read data (no IDs here; IDs are assigned by rank 0 during partitioning)
void read_CSV(const string &filename, vector<double> &X, vector<int> &Y, size_t &n_features)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    string line;
    getline(file, line); // discard first line

    stringstream ss(line);
    string cell;
    vector<string> fields;
    while (getline(ss, cell, ','))
        fields.push_back(cell);
    n_features = (fields.size() > 0 ? fields.size() - 1 : 0);

    while (getline(file, line))
    {
        stringstream ss2(line);
        string cell2;
        vector<string> fields2;
        while (getline(ss2, cell2, ','))
            fields2.push_back(cell2);

        if (fields2.size() < 2) // at least one feature + label
            continue;

        for (size_t i = 0; i + 1 < fields2.size(); ++i)
            X.push_back(stod(fields2[i]));

        int label = stoi(fields2.back());
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

// scale features (inplace)
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
            X[i * n_features + j] = (X[i * n_features + j] - min_val[j]) / range;
    }
}

// Gaussian kernel
double kernel(const double *v1, const double *v2, const size_t n_features)
{
    const double gamma = 0.00125;
    double res = 0.0;
    for (size_t i = 0; i < n_features; ++i)
    {
        double d = v1[i] - v2[i];
        res += d * d;
    }
    return exp(-gamma * res);
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

// calculate U, V
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
              vector<double> &alpha, double &b, double C = 10.0, bool init = true)
{
    const double eps = 1e-12;
    size_t n = y_train.size();
    vector<double> f(n, 0.0);
    
    if (init)
    {
        alpha.assign(n, 0.0);
        // alpha = 0 -> f[i] = -y[i]
        for (size_t i = 0; i < n; ++i)
            f[i] = -static_cast<double>(y_train[i]);
    }
    else
    {
        // compute f from current alpha
        for (size_t i = 0; i < n; ++i)
        {
            double sum = 0.0;
            const double *xi = x_train.data() + i * n_features;
            for (size_t j = 0; j < n; ++j)
            {
                if (alpha[j] == 0.0)
                    continue;
                const double *xj = x_train.data() + j * n_features;
                double kij = kernel(xj, xi, n_features);
                sum += alpha[j] * y_train[j] * kij;
            }
            f[i] = sum - static_cast<double>(y_train[i]);
        }
    }

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

    b = (b_high + b_low) / 2;
}

// void SMO_train(const vector<double> &x_train, const vector<int> &y_train, const size_t n_features,
//               vector<double> &alpha, double &b, double C = 10.0, bool init = true)
// {
//     const double eps = 1e-12;
//     size_t n = y_train.size();
//     vector<double> f(n, 0.0);

//     if (init)
//     {
//         // alpha = 0 -> f[i] = -y_train[i]
//         for (size_t i = 0; i < n; ++i)
//             f[i] = -static_cast<double>(y_train[i]);
//     }
//     else
//     {
//         // compute f from current alpha
//         for (size_t i = 0; i < n; ++i)
//         {
//             double sum = 0.0;
//             const double *xi = x_train.data() + i * n_features;
//             for (size_t j = 0; j < n; ++j)
//             {
//                 if (alpha[j] == 0.0)
//                     continue;
//                 const double *xj = x_train.data() + j * n_features;
//                 double kij = kernel(xj, xi, n_features);
//                 sum += alpha[j] * y_train[j] * kij;
//             }
//             f[i] = sum - static_cast<double>(y_train[i]);
//         }
//     }

//     // precompute kernel matrix
//     vector<double> K(n * n, 0.0);
//     for (size_t i = 0; i < n; ++i)
//     {
//         for (size_t j = i; j < n; ++j)
//         {
//             K[i * n + j] = kernel(x_train.data() + i * n_features, x_train.data() + j * n_features, n_features);
//             if (j != i)
//                 K[j * n + i] = K[i * n + j];
//         }
//     }

//     // init i_high / i_low
//     size_t i_high = calc_i_high(alpha, y_train, f, C);
//     size_t i_low = calc_i_low(alpha, y_train, f, C);

//     if (i_high == n || i_low == n)
//     {
//         b = 0.0;
//         return;
//     }

//     double b_high = f[i_high];
//     double b_low = f[i_low];

//     const double tau = 1e-5;
//     int num_iter = 0;
//     const int max_iter = 100000;

//     // use gap variable for loop
//     double gap = b_low - b_high;
//     while (gap > tau)
//     {
//         ++num_iter;
//         if (num_iter > max_iter)
//             break;

//         int s = y_train[i_high] * y_train[i_low];
//         double K11 = K[i_high * n + i_high];
//         double K22 = K[i_low * n + i_low];
//         double K12 = K[i_high * n + i_low];
//         double eta = K11 + K22 - 2.0 * K12;

//         double U, V;
//         if (!calculate_U_V(s, alpha[i_high], alpha[i_low], C, U, V))
//             break;

//         if (eta <= eps)
//             break;

//         double next_alpha_i_low = alpha[i_low] + y_train[i_low] * (b_high - b_low) / eta;
//         if (next_alpha_i_low > V) next_alpha_i_low = V;
//         if (next_alpha_i_low < U) next_alpha_i_low = U;

//         double next_alpha_i_high = alpha[i_high] + s * (alpha[i_low] - next_alpha_i_low);

//         double delta_alpha_i_high = next_alpha_i_high - alpha[i_high];
//         double delta_alpha_i_low = next_alpha_i_low - alpha[i_low];

//         for (size_t i = 0; i < n; ++i)
//         {
//             f[i] += delta_alpha_i_high * y_train[i_high] * K[i_high * n + i] + delta_alpha_i_low * y_train[i_low] * K[i_low * n + i];
//         }

//         alpha[i_high] = next_alpha_i_high;
//         alpha[i_low] = next_alpha_i_low;

//         i_high = calc_i_high(alpha, y_train, f, C);
//         i_low = calc_i_low(alpha, y_train, f, C);
//         if (i_high >= n || i_low >= n)
//             break;
//         b_high = f[i_high];
//         b_low = f[i_low];

//         gap = b_low - b_high;
//     }

//     b = (b_high + b_low) / 2.0;
// }

// Given alpha and tol, return indices of support vectors (alpha > tol)
vector<int> get_SV_indices(const vector<double> &alpha, double tol = 1e-8)
{
    vector<int> idx;
    for (size_t i = 0; i < alpha.size(); ++i)
        if (alpha[i] > tol)
            idx.push_back((int)i);
    return idx;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    int S = __builtin_ctz(world);

    if ((1u << S) != (unsigned)world)
    {
        if (rank == 0)
            cerr << "This cascade expects number of MPI ranks to be a power of two.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 0;
    }
    
    if(rank == 0) cout<<"[rank 0] Running CascadeSVM with "<<world << " processes"<<endl;

    // dataset name
    string dataset = "mnist3";
    // string dataset = "mnist2";
    // string dataset = "mnist";
    // string dataset = "banknote"; // C = 1, gamma = 0.125
    string train_file = dataset + "_train_data.csv";
    string test_file = dataset + "_test_data.csv";

    vector<double> x_all;
    vector<int> y_all;
    size_t n_features = 0;
    int n_total = 0;

    // rank 0 reads all data
    if (rank == 0)
    {
        read_CSV(train_file, x_all, y_all, n_features);
        n_total = (int)(y_all.size());
        if (n_total == 0)
        {
            cerr << "No training data read. Check file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast n_features and n_total
    unsigned long long nf_send = (unsigned long long)n_features;
    MPI_Bcast(&nf_send, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    n_features = (size_t)nf_send;
    MPI_Bcast(&n_total, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition data on rank 0 and send partitions; assign global IDs
    int n_parts = world;
    int chunk = (n_total + n_parts - 1) / n_parts;

    vector<double> x_part;
    vector<int> y_part;
    vector<int> id_part; // global IDs for local partition
    int my_n = 0;

    if (rank == 0)
    {
        for (int r = 0; r < n_parts; ++r)
        {
            int start = r * chunk;
            int end = min((r + 1) * chunk, n_total);
            int len = max(0, end - start);
            if (r == 0)
            {
                my_n = len;
                x_part.assign(x_all.begin() + start * n_features, x_all.begin() + (start + len) * n_features);
                y_part.assign(y_all.begin() + start, y_all.begin() + start + len);
                id_part.resize(len);
                for (int i = 0; i < len; ++i)
                    id_part[i] = start + i;
            }
            else
            {
                MPI_Send(&len, 1, MPI_INT, r, 10, MPI_COMM_WORLD);
                if (len > 0)
                {
                    MPI_Send(x_all.data() + start * n_features, len * (int)n_features, MPI_DOUBLE, r, 11, MPI_COMM_WORLD);
                    MPI_Send(y_all.data() + start, len, MPI_INT, r, 12, MPI_COMM_WORLD);
                    // send ids
                    vector<int> ids(len);
                    for (int i = 0; i < len; ++i)
                        ids[i] = start + i;
                    MPI_Send(ids.data(), len, MPI_INT, r, 13, MPI_COMM_WORLD);
                }
            }
        }
    }
    else
    {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        my_n = len;
        if (len > 0)
        {
            x_part.resize(len * n_features);
            y_part.resize(len);
            id_part.resize(len);
            MPI_Recv(x_part.data(), len * (int)n_features, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(y_part.data(), len, MPI_INT, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(id_part.data(), len, MPI_INT, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (rank == 0)
    {
        cout << "[rank 0] total samples = " << n_total << ", features = " << n_features << "\n";
    }
    
    chrono::high_resolution_clock::time_point t0, t1, t2;
    t0 = chrono::high_resolution_clock::now();

    // compute and broadcast global min/max using rank 0's data (x_all)
    vector<double> min_val(n_features, 0.0), max_val(n_features, 0.0);
    if (rank == 0)
    {
        find_min_max(x_all, min_val, max_val, n_total, n_features);
    }
    MPI_Bcast(min_val.data(), (int)n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(max_val.data(), (int)n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // scale partitions
    if (my_n > 0)
        scale_features(x_part, min_val, max_val, (size_t)my_n, n_features);

    // cascade parameters
    const double C = 10.0;
    const double tol_sv = 1e-8;
    const int max_rounds = 50;

    // SV_global stored on rank 0 and broadcast; each SV carries an ID
    vector<double> recv_X_sv;
    vector<int> recv_Y_sv;
    vector<double> recv_alpha_sv;
    vector<int> recv_ID_sv;
    int recv_sv_count = 0;

    vector<double> X_sv;
    vector<int> Y_sv;
    vector<double> alpha_sv;
    vector<int> ID_sv;
    int sv_count;

    double b_local = 0.0;
    vector<int> global_ID_sv; // the global ID of sv after each iteration, for convergence test

    int round = 0;
    bool converged = false;

    while (round < max_rounds)
    {
        // Next round
        ++round;

        if (rank == 0)
            cout << "=== Round " << round << " ===\n";
        
        // rank 0 broadcast its own fitted X_sv from last iteration to all other rank
        if(rank == 0)
        {
            recv_sv_count = (int) Y_sv.size();
            recv_X_sv = X_sv;
            recv_Y_sv = Y_sv;
            recv_alpha_sv = alpha_sv;
            recv_ID_sv = ID_sv;
        }
        else
        {
            recv_sv_count = 0;
        }

        MPI_Bcast(&recv_sv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (recv_sv_count > 0)
        {
            if (rank != 0)
            {
                recv_X_sv.resize((size_t)recv_sv_count * n_features);
                recv_Y_sv.resize(recv_sv_count);
                recv_alpha_sv.resize(recv_sv_count);
                recv_ID_sv.resize(recv_sv_count);
            }
            MPI_Bcast(recv_X_sv.data(), recv_sv_count * (int)n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(recv_Y_sv.data(), recv_sv_count, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(recv_alpha_sv.data(), recv_sv_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(recv_ID_sv.data(), recv_sv_count, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else
        {
            // ensure non-zero ranks have empty vectors
            if (rank != 0)
            {
                recv_X_sv.clear();
                recv_Y_sv.clear();
                recv_alpha_sv.clear();
                recv_ID_sv.clear();
            }
        }

        // initialize X_sv, Y_sv, ID_sv
        X_sv = x_part;
        Y_sv = y_part;
        ID_sv = id_part;
        sv_count = my_n;

        for (int step = 1; step <= world; step = 2*step)
        {
            if (rank % step == 0)
            {
                // Each rank constructs training set = local partition U SV_global (and tracks IDs) and deduplicates by ID
                // note: must deduplicates, otherwise there will be more and more vectors!!!!

                // initialize alpha_local as recv_alpha_sv, making use of previous training effort
                unordered_set<int> seen_ids;
                vector<double> X_train;
                vector<int> Y_train;
                vector<int> ID_train;
                vector<double> alpha_local;
                
                if (recv_sv_count > 0)
                {
                    X_train = recv_X_sv;
                    Y_train = recv_Y_sv;
                    alpha_local = recv_alpha_sv;
                    ID_train = recv_ID_sv;
                    seen_ids.insert(recv_ID_sv.begin(), recv_ID_sv.end());
                }

                for (int k = 0; k < sv_count; ++k)
                {
                    int gid = ID_sv[k];
                    if (seen_ids.insert(gid).second)
                    {
                        ID_train.push_back(gid);
                        Y_train.push_back(Y_sv[k]);
                        alpha_local.push_back(0.0);
                        for (size_t f = 0; f < n_features; ++f)
                            X_train.push_back(X_sv[k * n_features + f]);
                    }
                }

                // Train SMO on this augmented set
                if (X_train.size() > 0)
                {
                    SMO_train(X_train, Y_train, n_features, alpha_local, b_local, C, false);
                }
                else
                {
                    alpha_local.clear();
                    b_local = 0.0;
                }

                // Extract local support vectors (alpha > tol) and collect their IDs/features/labels/alphas
                vector<int> sv_idx = get_SV_indices(alpha_local, tol_sv);
                sv_count = (int)sv_idx.size();

                X_sv.resize((size_t)sv_count * n_features);
                Y_sv.resize(sv_count);
                alpha_sv.resize(sv_count);
                ID_sv.resize(sv_count);
                for (int k = 0; k < sv_count; ++k)
                {
                    int idx = sv_idx[k];
                    for (size_t f = 0; f < n_features; ++f)
                        X_sv[k * n_features + f] = X_train[(size_t)idx * n_features + f];
                    Y_sv[k] = Y_train[(size_t)idx];
                    alpha_sv[k] = alpha_local[(size_t)idx];
                    ID_sv[k] = ID_train[(size_t)idx]; // id corresponds to the source sample
                }
            }

            if (step < world)
            {
                if (rank % (2 * step) == step)
                {
                    int des = rank - step;
                    MPI_Send(&sv_count, 1, MPI_INT, des, 20, MPI_COMM_WORLD);
                    if (sv_count > 0)
                    {
                        MPI_Send(X_sv.data(), sv_count * (int)n_features, MPI_DOUBLE, des, 21, MPI_COMM_WORLD);
                        MPI_Send(Y_sv.data(), sv_count, MPI_INT, des, 22, MPI_COMM_WORLD);
                        MPI_Send(alpha_sv.data(), sv_count, MPI_DOUBLE, des, 23, MPI_COMM_WORLD);
                        MPI_Send(ID_sv.data(), sv_count, MPI_INT, des, 24, MPI_COMM_WORLD);
                    }
                }
                else if (rank % (2 * step) == 0)
                {
                    int src = rank + step;
                    MPI_Recv(&recv_sv_count, 1, MPI_INT, src, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    recv_X_sv.resize((size_t)recv_sv_count * n_features);
                    recv_Y_sv.resize(recv_sv_count);
                    recv_alpha_sv.resize(recv_sv_count);
                    recv_ID_sv.resize(recv_sv_count);
                    if (recv_sv_count > 0)
                    {
                        MPI_Recv(recv_X_sv.data(), recv_sv_count * (int)n_features, MPI_DOUBLE, src, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(recv_Y_sv.data(), recv_sv_count, MPI_INT, src, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(recv_alpha_sv.data(), recv_sv_count, MPI_DOUBLE, src, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(recv_ID_sv.data(), recv_sv_count, MPI_INT, src, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
        
        if (rank == 0)
        {
            // rank 0 first check convergence
            // if not convergent, then broadcast fitting result to all other processes
            // Convergence test: compare new set of IDs to previous global_ID_sv set (unordered)
            bool same = false;
            if (global_ID_sv.size() == ID_sv.size())
            {
                unordered_set<int> old_ids(global_ID_sv.begin(), global_ID_sv.end());
                bool all_in = true;
                for (int id : ID_sv)
                {
                    if (old_ids.find(id) == old_ids.end())
                    {
                        all_in = false;
                        break;
                    }
                }
                if (all_in)
                    same = true;
            }

            // update global_ID_sv
            global_ID_sv = ID_sv;

            // store the merged_b as the current global bias candidate
            double final_b_candidate = b_local;

            if (same)
            {
                cout << "[rank 0] Converged at round " << round << ", SV count = " << sv_count << "\n";
                converged = true;

                // final model: SV_global_* and final_b_candidate
                // Save final model files (IDs, labels, alphas, b)
                // ofstream fout_ids("final_sv_ids.txt");
                // ofstream fout_labels("final_sv_labels.txt");
                // ofstream fout_alphas("final_sv_alphas.txt");
                // ofstream fout_b("final_b.txt");
                // for (size_t i = 0; i < sv_count; ++i)
                // {
                //     fout_ids << ID_sv[i] << "\n";
                //     fout_labels << Y_sv[i] << "\n";
                //     fout_alphas << alpha_sv[i] << "\n";
                // }
                // fout_b << final_b_candidate << "\n";
                // fout_ids.close();
                // fout_labels.close();
                // fout_alphas.close();

                // cout << "[rank 0] Saved final SV ids/labels/alphas to files\n";
                cout << "[rank 0] Final b = "<< std::fixed << std::setprecision(15)  << final_b_candidate << endl;

                t1 = chrono::high_resolution_clock::now();
                
                // Run test prediction using final model
                vector<double> x_test;
                vector<int> y_test;
                size_t n_features_test = 0;
                read_CSV(test_file, x_test, y_test, n_features_test);
                if (n_features_test != n_features)
                {
                    cerr << "[rank 0] Warning: test n_features != train n_features\n";
                }
                size_t m = x_test.size() / n_features;
                if (m > 0)
                {
                    // scale test
                    scale_features(x_test, min_val, max_val, m, n_features);

                    // predict using SV_global
                    int correct = 0;
                    for (size_t i = 0; i < m; ++i)
                    {
                        double s = -final_b_candidate;
                        for (size_t k = 0; k < sv_count; ++k)
                        {
                            s += alpha_sv[k] * Y_sv[k] *
                                 kernel(x_test.data() + i * n_features, X_sv.data() + k * n_features, n_features);
                        }
                        int pred = (s >= 0 ? 1 : -1);
                        if (pred == y_test[i])
                            ++correct;
                    }
                    double acc = double(correct) / double(m);
                    cout << "[rank 0] Test accuracy (final model) = " << acc << " (" << correct << "/" << m << ")\n";
                }
                else
                {
                    cout << "[rank 0] No test data found or test file empty.\n";
                }
                
                t2 = chrono::high_resolution_clock::now();
            }
            else
            {
                cout << "[rank 0] Not converged yet. New SV count = " << sv_count << "\n";
                converged = false;
            }

        }
        
        // Broadcast convergence flag from rank 0 to all
        int conv_int = converged ? 1 : 0;
        MPI_Bcast(&conv_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        converged = (conv_int == 1);
        
        if (converged) break;
    }

    // timing
    if (rank == 0)
    {
        auto training_time = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        auto prediction_time = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(t2 - t0).count();
        cout << "[rank 0] Final global SV count = " << recv_Y_sv.size() << "\n";
        cout << "[rank 0] Cascade finished in " << round << " rounds\n";
        cout << "[rank 0] training time = " << training_time << " ms\n";
        cout << "[rank 0] prediction time = " << prediction_time << " ms\n";
        cout << "[rank 0] elapsed time = " << elapsed_time << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
