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
        size_t i_high = calc_i_high(alpha, y_train, f, C);
        size_t i_low = calc_i_low(alpha, y_train, f, C);
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

    // if (world != 4)
    // {
    //     if (rank == 0) cerr << "This program expects 4 MPI processes. Use mpirun -np 4\n";
    //     MPI_Finalize();
    //     return 1;
    // }
    if(rank == 0) cout<<"[rank 0] Running modified CascadeSVM with "<<world << " processes"<<endl;

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
    int my_start_id = 0; // global id of index 0 in my partition

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
                my_start_id = start;
                x_part.assign(x_all.begin() + start * n_features, x_all.begin() + (start + len) * n_features);
                y_part.assign(y_all.begin() + start, y_all.begin() + start + len);
                id_part.resize(len);
                for (int i = 0; i < len; ++i) id_part[i] = start + i;
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
                    for (int i = 0; i < len; ++i) ids[i] = start + i;
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
            my_start_id = id_part.size()>0 ? id_part[0] : 0;
        }
    }

    if (rank == 0) {
        cout << "[rank 0] total samples = " << n_total << ", features = " << n_features << "\n";
    }
    
    chrono::high_resolution_clock::time_point t0, t1, t2, tr_prev, tr;
    t0 = chrono::high_resolution_clock::now();
    tr_prev = t0;

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
    vector<double> SV_global_X; // flattened rows (n_sv * n_features)
    vector<int> SV_global_Y;
    vector<double> SV_global_alpha;
    vector<int> SV_global_ID;   // unique global id per SV

    int round = 0;
    bool converged = false;

    while (round < max_rounds && !converged)
    {
        if (rank == 0) cout << "=== Round " << round << " ===\n";

        // Rank 0 broadcasts current SV_global sizes and arrays
        int sv_count = (int)SV_global_Y.size();
        MPI_Bcast(&sv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (sv_count > 0)
        {
            if (rank != 0)
            {
                SV_global_X.resize((size_t)sv_count * n_features);
                SV_global_Y.resize(sv_count);
                SV_global_alpha.resize(sv_count);
                SV_global_ID.resize(sv_count);
            }
            MPI_Bcast(SV_global_X.data(), sv_count * (int)n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(SV_global_Y.data(), sv_count, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(SV_global_alpha.data(), sv_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(SV_global_ID.data(), sv_count, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else
        {
            // ensure non-zero ranks have empty vectors
            if (rank != 0)
            {
                SV_global_X.clear();
                SV_global_Y.clear();
                SV_global_alpha.clear();
                SV_global_ID.clear();
            }
        }

        // Each rank constructs training set = local partition U SV_global (and tracks IDs) and deduplicates by ID
        // note: must deduplicates, otherwise there will be more and more vectors!!!!
        vector<double> X_train;
        vector<int> Y_train;
        vector<double> alpha_local;
        double b_local = 0.0;
        vector<int> ID_train; // corresponding IDs for each row of X_train
        unordered_set<int> seen_ids;
        
        // initialize alpha_local as SV_global_alpha, making use of previous training effort
        if (sv_count > 0)
        {
            X_train = SV_global_X;
            Y_train = SV_global_Y;
            alpha_local = SV_global_alpha;
            ID_train = SV_global_ID;
            seen_ids.insert(SV_global_ID.begin(),SV_global_ID.end());
        }

        for(int k = 0; k< my_n; ++k)
        {
            int gid = id_part[k];
            if (seen_ids.insert(gid).second)
            {
                ID_train.push_back(gid);
                Y_train.push_back(y_part[k]);
                alpha_local.push_back(0.0);
                for (size_t f = 0; f < n_features; ++f)
                    X_train.push_back(x_part[k * n_features + f]);
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
        int local_sv_count = (int)sv_idx.size();

        vector<double> X_sv;
        vector<int> Y_sv;
        vector<double> alpha_sv;
        vector<int> ID_sv;
        if (local_sv_count > 0)
        {
            X_sv.resize((size_t)local_sv_count * n_features);
            Y_sv.resize(local_sv_count);
            alpha_sv.resize(local_sv_count);
            ID_sv.resize(local_sv_count);
            for (int k = 0; k < local_sv_count; ++k)
            {
                int idx = sv_idx[k];
                for (size_t f = 0; f < n_features; ++f)
                    X_sv[k * n_features + f] = X_train[(size_t)idx * n_features + f];
                Y_sv[k] = Y_train[(size_t)idx];
                alpha_sv[k] = alpha_local[(size_t)idx];
                ID_sv[k] = ID_train[(size_t)idx]; // id corresponds to the source sample
            }
        }

        // Send local SVs to rank 0; rank 0 gathers and deduplicates by ID
        if (rank == 0)
        {
            // merged containers (unique by ID)
            vector<double> merged_X;
            vector<int> merged_Y;
            vector<double> merged_alpha;
            double merged_b = 0.0;
            vector<int> merged_ID;
            unordered_set<int> seen_ids;

            // first include rank 0's own SVs
            if (local_sv_count > 0)
            {
                merged_X = X_sv;
                merged_Y = Y_sv;
                merged_alpha = alpha_sv;
                merged_ID = ID_sv;
                seen_ids.insert(ID_sv.begin(),ID_sv.end());
            }
            // if (local_sv_count > 0)
            // {
            //     for (int k = 0; k < local_sv_count; ++k)
            //     {
            //         int gid = ID_sv[k];
            //         if (seen_ids.insert(gid).second)
            //         {
            //             // new
            //             merged_ID.push_back(gid);
            //             merged_Y.push_back(Y_sv[k]);
            //             merged_alpha.push_back(alpha_sv[k]);
            //             for (size_t f = 0; f < n_features; ++f)
            //                 merged_X.push_back(X_sv[k * n_features + f]);
            //         }
            //     }
            // }

            // receive from other ranks
            for (int src = 1; src < world; ++src)
            {
                int recv_count = 0;
                MPI_Recv(&recv_count, 1, MPI_INT, src, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (recv_count > 0)
                {
                    vector<double> Xbuf((size_t)recv_count * n_features);
                    vector<int> Ybuf(recv_count);
                    vector<double> abuf(recv_count);
                    vector<int> idbuf(recv_count);
                    MPI_Recv(Xbuf.data(), recv_count * (int)n_features, MPI_DOUBLE, src, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(Ybuf.data(), recv_count, MPI_INT, src, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(abuf.data(), recv_count, MPI_DOUBLE, src, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(idbuf.data(), recv_count, MPI_INT, src, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (int k = 0; k < recv_count; ++k)
                    {
                        int gid = idbuf[k];
                        if (seen_ids.insert(gid).second)
                        {
                            merged_ID.push_back(gid);
                            merged_Y.push_back(Ybuf[k]);
                            // merged_alpha.push_back(abuf[k]);
                            merged_alpha.push_back(0.0);
                            for (size_t f = 0; f < n_features; ++f)
                                merged_X.push_back(Xbuf[k * n_features + f]);
                        }
                    }
                }
            }

            int merged_count = (int)merged_ID.size();
            cout << "[rank 0] merged unique SV count from workers = " << merged_count << "\n";
            
            // Train SMO on merged unique SVs (to get new global SV set)
            if (merged_count > 0)
            {
                SMO_train(merged_X, merged_Y, n_features, merged_alpha, merged_b, C, false);
            }
            else
            {
                merged_alpha.clear();
                merged_b = 0.0;
            }

            // extract SVs from merged training
            vector<int> new_sv_idx = get_SV_indices(merged_alpha, tol_sv);
            int new_sv_count = (int)new_sv_idx.size();
            vector<double> SV_new_X;
            vector<int> SV_new_Y;
            vector<double> SV_new_alpha;
            vector<int> SV_new_ID;
            SV_new_X.reserve((size_t)new_sv_count * n_features);
            SV_new_Y.reserve(new_sv_count);
            SV_new_alpha.reserve(new_sv_count);
            SV_new_ID.reserve(new_sv_count);

            for (int k = 0; k < new_sv_count; ++k)
            {
                int id_in_merged = new_sv_idx[k];
                SV_new_ID.push_back(merged_ID[id_in_merged]);
                SV_new_Y.push_back(merged_Y[id_in_merged]);
                SV_new_alpha.push_back(merged_alpha[id_in_merged]);
                for (size_t f = 0; f < n_features; ++f)
                    SV_new_X.push_back(merged_X[(size_t)id_in_merged * n_features + f]);
            }

            // Convergence test: compare new set of IDs to previous SV_global_ID set (unordered)
            bool same = false;
            if (SV_global_ID.size() == SV_new_ID.size())
            {
                unordered_set<int> old_ids(SV_global_ID.begin(), SV_global_ID.end());
                bool all_in = true;
                for (int id : SV_new_ID)
                {
                    if (old_ids.find(id) == old_ids.end())
                    {
                        all_in = false;
                        break;
                    }
                }
                if (all_in) same = true;
            }

            // update SV_global with SV_new (swap to avoid copy)
            SV_global_X.swap(SV_new_X);
            SV_global_Y.swap(SV_new_Y);
            SV_global_alpha.swap(SV_new_alpha);
            SV_global_ID.swap(SV_new_ID);

            // store the merged_b as the current global bias candidate
            double final_b_candidate = merged_b;
            
            tr = chrono::high_resolution_clock::now();
            if (rank == 0)
            {
                auto round_time = chrono::duration_cast<chrono::milliseconds>(tr - tr_prev).count();
                cout << "[rank 0] Round" << round <<" takes " << round_time << " ms\n";
            }
            tr_prev = tr;

            if (same)
            {
                cout << "[rank 0] Converged at round " << round << ", SV count = " << SV_global_Y.size() << "\n";
                converged = true;

                // final model: SV_global_* and final_b_candidate
                // Save final model files (IDs, labels, alphas, b)
                // ofstream fout_ids("final_sv_ids.txt");
                // ofstream fout_labels("final_sv_labels.txt");
                // ofstream fout_alphas("final_sv_alphas.txt");
                // ofstream fout_b("final_b.txt");
                // for (size_t i = 0; i < SV_global_ID.size(); ++i)
                // {
                //     fout_ids << SV_global_ID[i] << "\n";
                //     fout_labels << SV_global_Y[i] << "\n";
                //     fout_alphas << SV_global_alpha[i] << "\n";
                // }
                // fout_b << final_b_candidate << "\n";
                // fout_ids.close(); fout_labels.close(); fout_alphas.close();

                // cout << "[rank 0] Saved final SV ids/labels/alphas to files\n";
                cout << "[rank 0] Final b = "<< std::fixed << std::setprecision(15) << final_b_candidate <<endl;
                
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
                        for (size_t k = 0; k < SV_global_Y.size(); ++k)
                        {
                            s += SV_global_alpha[k] * SV_global_Y[k] *
                                 kernel(x_test.data() + i * n_features, SV_global_X.data() + k * n_features, n_features);
                        }
                        int pred = (s >= 0 ? 1 : -1);
                        if (pred == y_test[i]) ++correct;
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
                cout << "[rank 0] Not converged yet. New SV count = " << SV_global_Y.size() << "\n";
                converged = false;
            }
        }
        else
        {
            // Non-zero ranks send their SVs (count, X, Y, alpha, ID) to rank 0
            MPI_Send(&local_sv_count, 1, MPI_INT, 0, 20, MPI_COMM_WORLD);
            if (local_sv_count > 0)
            {
                MPI_Send(X_sv.data(), local_sv_count * (int)n_features, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
                MPI_Send(Y_sv.data(), local_sv_count, MPI_INT, 0, 22, MPI_COMM_WORLD);
                MPI_Send(alpha_sv.data(), local_sv_count, MPI_DOUBLE, 0, 23, MPI_COMM_WORLD);
                MPI_Send(ID_sv.data(), local_sv_count, MPI_INT, 0, 24, MPI_COMM_WORLD);
            }
            // then wait for next broadcast
        }

        // Broadcast convergence flag from rank 0 to all
        int conv_int = converged ? 1 : 0;
        MPI_Bcast(&conv_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        converged = (conv_int == 1);
        
        // Next round
        ++round;
    }

    // timing
    if (rank == 0)
    {
        auto training_time = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        auto prediction_time = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(t2 - t0).count();
        cout << "[rank 0] Final global SV count = " << SV_global_Y.size() << "\n";
        cout << "[rank 0] Cascade finished in " << round << " rounds\n";
        cout << "[rank 0] training time = " << training_time << " ms\n";
        cout << "[rank 0] prediction time = " << prediction_time << " ms\n";
        cout << "[rank 0] elapsed time = " << elapsed_time << " ms\n";
    }

    MPI_Finalize();
    return 0;
}