#include <iostream>
#include <fstream> // for opening files
#include <vector>
#include <limits>    // for defining infinity
#include <cmath>     // for power function
#include <math.h>
#include <algorithm> // for std::max and std::min
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
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

// calculate min value, max value of each feature
// blockDim(32,32)
// GridDim((n+31)/32, (n_features+31)/32)
__global__ void find_min_max(const double *min_val_prev, const double *max_val_prev,  double *min_val, double *max_val, int n_samples, int n_features)
{
    int local_j = threadIdx.x;
    int local_i = threadIdx.y;
    int tid = local_i*blockDim.x + local_j;
    
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int gid = i*n_features + j;  // global ID
    
    //note that the Max_copy and Min_copy need memory allocation when calling kernel function
    extern __shared__ double shmem[];
    double *Max_copy = shmem;
    double *Min_copy = shmem + blockDim.x*blockDim.y;
    Min_copy[tid] = (j < n_features) ? ((i<n_samples)? min_val_prev[gid] : min_val_prev[(n_samples-1)*n_features + j]) : 0.0;
    Max_copy[tid] = (j < n_features) ? ((i<n_samples)? max_val_prev[gid] : max_val_prev[(n_samples-1)*n_features + j]) : 0.0;
    __syncthreads(); // must synchronize here, otherwise there are errors
    
    for(int s = blockDim.y>>1; s >0; s>>=1){
        if(local_i < s)
        {
            Min_copy[tid] = fmin(Min_copy[tid], Min_copy[tid + s*blockDim.x]);
            Max_copy[tid] = fmax(Max_copy[tid], Max_copy[tid + s*blockDim.x]);
        }
        __syncthreads();
    }
    // need to make sure blockDim is a power of 2!!!
    
    if(local_i ==0 && j < n_features)
    {
        min_val[blockIdx.y*n_features + j] = Min_copy[local_j];
        max_val[blockIdx.y*n_features + j] = Max_copy[local_j];
    }
}

// scale features
// blockDim(32,32)
// GridDim((n_samples+31)/32, (n_features+31)/32)
__global__ void scale_features(double *X,
                    const double *min_val, const double *max_val,
                    const size_t n_samples, const size_t n_features)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int gid = i*n_features + j;  // global ID
    
    if(i < n_samples && j < n_features)
    {
        double range = max_val[j] - min_val[j];
        if (range < 1e-12)
            range = 1.0;
        
        X[gid] = (X[gid] - min_val[j]) / range;
    }
}

// define kernel function
__device__ double kernel(const double *v1, const double *v2, const size_t n_features)
{
    // Gaussian kernel
    const double gamma = 0.00125;
    double res = 0.0;
    for (size_t i = 0; i < n_features; ++i)
    {
        // res += pow(v1[i] - v2[i], 2.0);
        res += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }
    res = exp(-gamma * res);

    return res;
}

// calculate kernel maxtrix
// blockDim(32,32)
// GridDim((n1+31)/32, (n2+31)/32)
__global__ void calc_kernel_matrix(double *K, const double *X1, const double *X2, const size_t n1, const size_t n2, const size_t n_features)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int gid = i*n2 + j;  // global ID
    
    if(i < n1 && j < n2)
    {
        K[gid] = kernel(X1 + i*n_features, X2 + j*n_features, n_features);
    }
}

// initialize alpha and f
// blockDim(1024)
//gridDim((n_samples + 1023)/1024)
__global__ void init_alpha_f(double *alpha, double *f, const int *y, const size_t n_samples)
{
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < n_samples) 
    {
        alpha[gid] = 0;
        f[gid] = -y[gid];
    }
}

// calculate f_in_I_high and initial i_high
// blockDim(1024)
// GridDim((n_samples+1023)/1024)
__global__ void calc_f_in_I_high(double *f_in_I_high, size_t *i_high, const double *alpha, const int *y, const double *f, const size_t n_samples, const double C, const double eps)
{
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < n_samples)
    {
        bool in_I_high = ((y[gid] == 1 && alpha[gid] < C - eps) || (y[gid] == -1 && alpha[gid] > 0.0 + eps));
        f_in_I_high[gid] = in_I_high ? f[gid] : INFINITY;
        i_high[gid] = gid;
    }
}

// calculate i_high
// blockDim(1024)
// GridDim((n+1023)/1024)
// n should be the size of i_high_prev
__global__ void calc_i_high(const double *f, const size_t n, size_t *i_high_prev, size_t *i_high)
{
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;

    for(int s = blockDim.x>>1; s >0; s>>=1){
        if( tid < s && gid + s < n)
        {
            if(f[i_high_prev[gid + s]] < f[i_high_prev[gid]]) i_high_prev[gid] = i_high_prev[gid + s];
        }
        __syncthreads();
    }
    // need to make sure blockDim is a power of 2!!!
    
    if(tid ==0)
    {
        i_high[blockIdx.x] = i_high_prev[gid];
    }
}

// calculate f_in_I_low and initial i_low
// blockDim(1024)
// GridDim((n_samples+1023)/1024)
__global__ void calc_f_in_I_low(double *f_in_I_low, size_t *i_low, const double *alpha, const int *y, const double *f, const size_t n_samples, const double C, const double eps)
{
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < n_samples)
    {
        bool in_I_low = ((y[gid] == 1 && alpha[gid] > 0.0 + eps) || (y[gid] == -1 && alpha[gid] < C - eps));
        f_in_I_low[gid] = in_I_low ? f[gid] : -INFINITY;
        i_low[gid] = gid;
    }
}

// calculate i_low
// blockDim(1024)
// GridDim((n+1023)/1024)
// n should be the size of i_low_prev
__global__ void calc_i_low(const double *f, const size_t n, size_t *i_low_prev, size_t *i_low)
{
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;

    for(int s = blockDim.x>>1; s >0; s>>=1){
        if( tid < s && gid + s < n)
        {
            if(f[i_low_prev[gid + s]] > f[i_low_prev[gid]]) i_low_prev[gid] = i_low_prev[gid + s];
        }
        __syncthreads();
    }
    // need to make sure blockDim is a power of 2!!!
    
    if(tid ==0)
    {
        i_low[blockIdx.x] = i_low_prev[gid];
    }
}

// calculate lower bound, U, and upper bound, V, for alpha_i_low
bool calculate_U_V(const int s, const double alpha_i_high, const double alpha_i_low, const double C,
                   double &U, double &V)
{
    if (s == -1)
    {
        U = fmax(0.0, alpha_i_low - alpha_i_high);
        V = fmin(C, C + alpha_i_low - alpha_i_high);
    }
    else
    {
        U = fmax(0.0, alpha_i_low + alpha_i_high - C);
        V = fmin(C, alpha_i_low + alpha_i_high);
    }
    return U <= V + 1e-12;
}


// update f
// blockDim(1024)
// gridDim((n_samples + 1023)/1024)
__global__ void update_f(double *f, const double *K_i_high, const double *K_i_low, const int *y_train, 
const double delta_alpha_i_high, const double delta_alpha_i_low, 
const size_t n_samples, const size_t i_high, const size_t i_low)
{
     size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < n_samples) 
    {
        f[gid] += delta_alpha_i_high * y_train[i_high] * K_i_high[gid] + delta_alpha_i_low * y_train[i_low] * K_i_low[gid];
    }
}

// predict test data
// blockDim(1024)
// gridDim((m + 1023)/1024)
__global__ void predict(double *accuracy, const int*y_test, const int *y_train, const double *x_test, const double *x_train, 
                        const double *alpha, const size_t m, const size_t n, const size_t n_features, const double b)
{
     size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
     if(gid >= m) return;
     
     const double* x_test_i = x_test + gid*n_features;
     double f = -b; // note that the prediction function is sign(wx -b)
     for (size_t j = 0; j < n; ++j)
     {
        const double* x_train_j = x_train + j*n_features;
        double k = kernel(x_test_i, x_train_j, n_features);
        f += alpha[j] * y_train[j] * k;
     }
     int y_pred = (f > 0) ? 1 : -1;
     accuracy[gid] = y_pred == y_test[gid];
}

//calculate sum
// blockDim(1024)
// gridDim((m + 1023)/1024)
// m is the size of B
__global__ void reduce_sum(double *B, double *C, int m)
{
    size_t tid = threadIdx.x; // thread ID 
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x; // global ID; Note that block.Dim must be larger than 1
    
    extern __shared__ double B_copy[];
    B_copy[tid] = (gid < m) ? B[gid]: 0.0; // B_copy store B for a block
    __syncthreads();
    
    for(int s = blockDim.x>>1; s > 0; s>>=1){
        if(tid < s) B_copy[tid] += B_copy[tid + s];
        __syncthreads();
    }
    // need to make sure blockDim is a power of 2!!!
    
    if(tid == 0) C[blockIdx.x] = B_copy[0];
}

// train data using SMO algorithm
__host__ void SMO_train(const double * dev_x_train, const int *dev_y_train, const vector<int> &y_train, const size_t n, const size_t n_features,
               double * dev_alpha, double &b, double C = 10.0)
{
    const double eps = 1e-12;
    double *dev_f;
    cudaError_t err = cudaMalloc(&dev_f, n*sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }
    
    // int threadnum1 = 32;
    int threadnum2 = 1024;
    size_t row_num;
    
    // initialize dev_alpha and dev_f
    size_t blocknum = (n + threadnum2-1)/threadnum2;
    init_alpha_f<<<blocknum, threadnum2>>>(dev_alpha, dev_f, dev_y_train, n);

    // choose i_high, i_low
    double *dev_f_in_I_high;
    double *dev_f_in_I_low;
    size_t *dev_i_high_prev, *dev_i_high;
    size_t *dev_i_low_prev, *dev_i_low;
    cudaMalloc(&dev_f_in_I_high, n*sizeof(double));
    cudaMalloc(&dev_f_in_I_low, n*sizeof(double));
    cudaMalloc(&dev_i_high_prev, n*sizeof(size_t));
    cudaMalloc(&dev_i_high, n*sizeof(size_t));
    cudaMalloc(&dev_i_low_prev, n*sizeof(size_t));
    cudaMalloc(&dev_i_low, n*sizeof(size_t));
    
    size_t i_high;
    size_t i_low;
    double b_high;
    double b_low;
    
    size_t i_high_prev = n;          // only for updating K_i_high
    size_t i_low_prev = n;           // only for updating K_i_low
    double *dev_K_i_high; // the row contain K(i_high, j) for j = 0,..n-1
    double *dev_K_i_low;  // the row contain K(i_low, j) for j = 0,..n-1
    cudaMalloc(&dev_K_i_high, n*sizeof(double));
    cudaMalloc(&dev_K_i_low, n*sizeof(double));

    const double tau = 1e-5;
    int num_iter = 1;
    const int max_iter = 100000;
    while (true)
    {
        // update i_high, i_low, b_high, b_low
        calc_f_in_I_high<<<blocknum, threadnum2>>>(dev_f_in_I_high, dev_i_high, dev_alpha, dev_y_train, dev_f, n, C, eps);
        calc_f_in_I_low<<<blocknum, threadnum2>>>(dev_f_in_I_low, dev_i_low, dev_alpha, dev_y_train, dev_f, n, C, eps);
        
        row_num = n;
        blocknum = (row_num+threadnum2-1)/threadnum2;
        while(row_num > 1)
        {
            swap(dev_i_high_prev, dev_i_high);
            swap(dev_i_low_prev, dev_i_low);
            calc_i_high<<<blocknum, threadnum2>>>(dev_f_in_I_high, row_num, dev_i_high_prev, dev_i_high);
            calc_i_low<<<blocknum, threadnum2>>>(dev_f_in_I_low, row_num, dev_i_low_prev, dev_i_low);
            row_num = blocknum;
            blocknum = (row_num+threadnum2-1)/threadnum2;
        }
        
        cudaMemcpy(&i_high, dev_i_high, sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&i_low, dev_i_low, sizeof(size_t), cudaMemcpyDeviceToHost);
        
        if (i_high >= n || i_low >= n)
        {
            cerr << "i_high or i_low not found; iteration stops" << endl;
            break;
        }
        cudaMemcpy(&b_high, dev_f+i_high, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b_low, dev_f+i_low, sizeof(double), cudaMemcpyDeviceToHost);
        
        if(b_low <= b_high + 2.0 * tau) break;

        blocknum = (n+threadnum2-1)/threadnum2;
        if (i_high != i_high_prev)
        {
            // update i_high_prev and K_i_high
            // in the first iteration, i_high_prev is set to n, so i_high != i_high_prev, there must be an update
            i_high_prev = i_high;
            calc_kernel_matrix<<<blocknum, threadnum2>>>(dev_K_i_high, dev_x_train + i_high*n_features, dev_x_train, 1, n, n_features);
            cudaDeviceSynchronize();
        }

        if (i_low != i_low_prev)
        {
            // update i_low_prev and K_i_low
            // in the first iteration, i_low_prev is set to n, so i_low != i_low_prev, there must be an update
            i_low_prev = i_low;
            calc_kernel_matrix<<<blocknum, threadnum2>>>(dev_K_i_low, dev_x_train + i_low*n_features, dev_x_train, 1, n, n_features);
            cudaDeviceSynchronize();
        }

        // update alpha
        int s = y_train[i_high] * y_train[i_low];
        double K11, K22, K12;
        
        cudaMemcpy(&K11, dev_K_i_high + i_high, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&K22, dev_K_i_low + i_low, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&K12, dev_K_i_high + i_low, sizeof(double), cudaMemcpyDeviceToHost);
        
        double eta = K11 + K22 - 2.0 * K12;
        double alpha_i_high;
        double alpha_i_low;
        cudaMemcpy(&alpha_i_high, dev_alpha + i_high, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&alpha_i_low, dev_alpha + i_low, sizeof(double), cudaMemcpyDeviceToHost);

        double U, V;
        if (!calculate_U_V(s, alpha_i_high, alpha_i_low, C, U, V))
        {
            cerr << "warning: infeasible U and V; iteration stops" << endl;
            break;
        }

        double next_alpha_i_low = alpha_i_low;
        if (eta <= eps)
        {
            cerr << "warning: non positive eta; iteration stops" << endl;
            break;
        }

        next_alpha_i_low = alpha_i_low + y_train[i_low] * (b_high - b_low) / eta;
        // clip to [U, V]
        if (next_alpha_i_low > V)
            next_alpha_i_low = V;
        if (next_alpha_i_low < U)
            next_alpha_i_low = U;

        double next_alpha_i_high = alpha_i_high + s * (alpha_i_low - next_alpha_i_low);

        // update dev_f on device
        double delta_alpha_i_high = next_alpha_i_high - alpha_i_high;
        double delta_alpha_i_low = next_alpha_i_low - alpha_i_low;
        blocknum = (n+threadnum2-1)/threadnum2;
        update_f<<<blocknum, threadnum2>>>(dev_f, dev_K_i_high, dev_K_i_low, dev_y_train, delta_alpha_i_high, delta_alpha_i_low, n, i_high, i_low);

        // updtae dev_alpha
        cudaMemcpy(dev_alpha + i_high, &next_alpha_i_high, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_alpha + i_low, &next_alpha_i_low, sizeof(double), cudaMemcpyHostToDevice);
        
        ++num_iter;
        // avoid infinite loop
        if (num_iter > max_iter)
        {
            cerr << "Too many iterations; stopping" << endl;
            break;
        }
    }

    cudaFree(dev_f_in_I_high);
    cudaFree(dev_f_in_I_low);
    cudaFree(dev_i_high_prev);
    cudaFree(dev_i_high);
    cudaFree(dev_i_low_prev);
    cudaFree(dev_i_low);
    cudaFree(dev_f);
    cudaFree(dev_K_i_high);
    cudaFree(dev_K_i_low);

    cout << "number of iterations: " << num_iter << endl;
    b = (b_high + b_low) / 2;
    cout << "b = "<< std::fixed << std::setprecision(15)  << b << endl;
    cout << "(b_high - b_low)/2*1e10 = " << (b_high - b_low)/2*1e10 << endl;
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
    cout<<"n = "<<n <<endl;
    cout<<"n_features = "<< n_features <<endl;
    if (n == 0)
    {
        cerr << "Error: No data read from file." << endl;
        return 1;
    }
    
    // cout<<"y_train: "<<endl;
    // for(size_t i = 0; i<n; i++)
    // {
    //     cout<<y_train[i]<<endl;
    // }

    //initialize the CUDA timer
    float training_time = 0;
    float prediction_time = 0;
    float elapsed_time = 0;
    cudaEvent_t start, stop1, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    
    cudaEventRecord(start);

    cudaError_t err;
    double *dev_x_train;
    int *dev_y_train;
    double *dev_min_val_prev;
    double *dev_min_val;
    double *dev_max_val_prev;
    double *dev_max_val;
    
    const int threadnum1 = 32;
    const int threadnum2 = 1024;
    
    cudaMalloc(&dev_x_train, n*n_features*sizeof(double));
    cudaMalloc(&dev_y_train, n*sizeof(int));
    cudaMalloc(&dev_min_val_prev, n*n_features*sizeof(double));
    cudaMalloc(&dev_min_val, n*n_features*sizeof(double));
    cudaMalloc(&dev_max_val_prev, n*n_features*sizeof(double));
    cudaMalloc(&dev_max_val, n*n_features*sizeof(double));
    
    cudaMemcpy(dev_x_train, x_train.data(), n*n_features*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_train, y_train.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_min_val, x_train.data(), n*n_features*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max_val, x_train.data(), n*n_features*sizeof(double), cudaMemcpyHostToDevice);
    
    // cudaMemcpy(y_train.data(), dev_y_train, n*sizeof(int), cudaMemcpyDeviceToHost);
    // cout<<"y_train: "<<endl;
    // for(size_t i = 0; i<n; i++)
    // {
    //     cout<<y_train[i]<<endl;
    // }
    
    // find min and max values of each features
    size_t row_num = n;
    dim3 blockDim(threadnum1,threadnum1);
    dim3 gridDim;
    gridDim.x = (n_features+threadnum1-1)/threadnum1;
    gridDim.y = (row_num+threadnum1-1)/threadnum1;

    while(row_num > 1)
    {
        swap(dev_min_val_prev, dev_min_val);
        swap(dev_max_val_prev, dev_max_val);
        size_t per_block_shmem = 2*threadnum2 * sizeof(double); 
        find_min_max<<<gridDim, blockDim, per_block_shmem>>>(dev_min_val_prev, dev_max_val_prev, dev_min_val, dev_max_val, row_num, n_features);
        row_num = gridDim.y;
        gridDim.y = (row_num+threadnum1-1)/threadnum1;
    }
    
    // store min and max values 
    double *dev_feature_min;
    double *dev_feature_max;
    err = cudaMalloc(&dev_feature_min, n_features*sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&dev_feature_max, n_features*sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(dev_feature_min, dev_min_val, n_features*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_feature_max, dev_max_val, n_features*sizeof(double), cudaMemcpyDeviceToDevice);
    
    cudaFree(dev_min_val_prev);
    cudaFree(dev_min_val);
    cudaFree(dev_max_val_prev);
    cudaFree(dev_max_val);
    
    // scale train data
    gridDim.x = (n_features+threadnum1-1)/threadnum1;
    gridDim.y = (n+threadnum1-1)/threadnum1;
    scale_features<<<gridDim, blockDim>>>(dev_x_train, dev_feature_min, dev_feature_max, n, n_features);
    cudaDeviceSynchronize();

    // hyperparameters
    double C = 10.0;
    double *dev_alpha;
    err = cudaMalloc(&dev_alpha, n*sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }
    double b;
    
    SMO_train(dev_x_train, dev_y_train, y_train, n, n_features, dev_alpha, b, C);
    
    //record training time
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    
    //calculate training time
    cudaEventElapsedTime(&training_time, start, stop1);
    
    // read test dataset
    string test_file = dataset + "_test_data.csv";
    vector<double> x_test;
    vector<int> y_test;
    read_CSV(test_file, x_test, y_test, n_features);
    size_t m = x_test.size() / n_features;
    
    // cout<<"m: "<<m<<endl;
    // cout<<"n_features: "<<n_features<<endl;
    
    double *dev_x_test;
    int *dev_y_test;
    
    cudaMalloc(&dev_x_test, m*n_features*sizeof(double));
    cudaMalloc(&dev_y_test, m*sizeof(int));
    
    cudaMemcpy(dev_x_test, x_test.data(), m*n_features*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_test, y_test.data(), m*sizeof(int), cudaMemcpyHostToDevice);
    
    gridDim.x = (n_features+threadnum1-1)/threadnum1;
    gridDim.y = (m+threadnum1-1)/threadnum1;
    scale_features<<<gridDim, blockDim>>>(dev_x_test, dev_feature_min, dev_feature_max, m, n_features);
    cudaDeviceSynchronize();
    
    cudaFree(dev_feature_min);
    cudaFree(dev_feature_max);
    
    // make prediction
    double *dev_accuracy_prev;
    double *dev_accuracy;
    cudaMalloc(&dev_accuracy_prev, m*sizeof(double));
    cudaMalloc(&dev_accuracy, m*sizeof(double));
    
    size_t blocknum = (m+threadnum2-1)/threadnum2;
    predict<<<blocknum, threadnum2>>>(dev_accuracy, dev_y_test, dev_y_train, dev_x_test, dev_x_train, dev_alpha, m, n, n_features, b);
    cudaDeviceSynchronize();
    
    // calculate accuracy
    row_num = m;
    blocknum = (row_num+threadnum2-1)/threadnum2;
    while(row_num > 1){
        std::swap(dev_accuracy_prev, dev_accuracy);
        reduce_sum<<<blocknum, threadnum2, threadnum2 * sizeof(double)>>>(dev_accuracy_prev, dev_accuracy, row_num);
        row_num = blocknum; //how many elements remaining to sum up
        blocknum = (row_num + threadnum2 -1)/threadnum2;
    }

    double accuracy;
    cudaMemcpy(&accuracy, dev_accuracy, sizeof(double), cudaMemcpyDeviceToHost);
    int correct = (int) accuracy;
    accuracy /= m;
    cout << "Test accuracy = " << std::fixed << std::setprecision(15) << accuracy << " (" << correct << "/" << m << ")\n";
    
    vector<double> alpha(n,0.0);
    cudaMemcpy(alpha.data(), dev_alpha, n*sizeof(double), cudaMemcpyDeviceToHost);
    int sv_count = 0;
    double tol = 1e-8;
    for (size_t j = 0; j < n; ++j)
    {
        if (alpha[j] > tol)
            sv_count++;
    }
    cout << "Final SV count = " << sv_count << "\n";

    //stop the CUDA timer
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    
    //calculate prediction time
    cudaEventElapsedTime(&prediction_time, stop1, stop2);
    
    //calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop2);

    cout<<"The training time: "<<training_time<<" milliseconds"<<endl;
    cout<<"The prediction time: "<<prediction_time<<" milliseconds"<<endl;
    cout<<"The elapsed time: "<<elapsed_time<<" milliseconds"<<endl;
    
    cudaFree(dev_x_train);
    cudaFree(dev_y_train);
    
    cudaFree(dev_alpha);
    
    cudaFree(dev_x_test);
    cudaFree(dev_y_test);
    
    cudaFree(dev_accuracy_prev);
    cudaFree(dev_accuracy);

    return 0;
}