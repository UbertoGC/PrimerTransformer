#include "Matriz2D.h"

#define BLOCK_SIZE 16

// ============================================
// Kernel: Multiplicación de matrices (float)
// ============================================
__global__ void matMulKernel(const float* A, const float* B, float* C,
                             int filasA, int colsA, int colsB) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float val = 0.0f;

    for (int t = 0; t < (colsA + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < filasA && (t * BLOCK_SIZE + threadIdx.x) < colsA)
            tileA[threadIdx.y][threadIdx.x] = A[row * colsA + t * BLOCK_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < colsB && (t * BLOCK_SIZE + threadIdx.y) < colsA)
            tileB[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * colsB + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < filasA && col < colsB)
        C[row * colsB + col] = val;
}

// ============================================
// Kernel: ReLU
// ============================================
__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

// ============================================
// Kernel: Softmax por filas
// ============================================
__global__ void softmaxKernel(float* A, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;
    float* rowData = A + row * cols;

    // Max
    float maxVal = -1e30f;
    for (int j = tid; j < cols; j += blockDim.x)
        maxVal = fmaxf(maxVal, rowData[j]);
    shared[tid] = maxVal;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    maxVal = shared[0];

    // Exp y suma
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        rowData[j] = expf(rowData[j] - maxVal);
        sum += rowData[j];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    sum = shared[0];

    for (int j = tid; j < cols; j += blockDim.x)
        rowData[j] /= sum;
}

// ============================================
// Kernel: Normalización por filas
// ============================================
__global__ void normalizeKernel(float* A, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* mean = shared;
    float* var = shared + 1;

    if (threadIdx.x == 0) {
        *mean = 0.0f;
        *var = 0.0f;
    }
    __syncthreads();

    atomicAdd(mean, A[row * cols + threadIdx.x]);
    __syncthreads();

    if (threadIdx.x == 0) *mean /= cols;
    __syncthreads();

    float diff = A[row * cols + threadIdx.x] - *mean;
    atomicAdd(var, diff * diff);
    __syncthreads();

    if (threadIdx.x == 0) *var = sqrtf(*var / cols);
    __syncthreads();

    if (*var > 0)
        A[row * cols + threadIdx.x] = diff / *var;
}

// ============================================
// Wrappers CUDA en Matriz2D
// ============================================
void Matriz2D::RELU_CUDA() {
    size_t size = filas * columnas;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

void Matriz2D::SoftmaxFilas_CUDA() {
    size_t size = filas * columnas;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    size_t shared_mem = threads * sizeof(float);
    softmaxKernel<<<filas, threads, shared_mem>>>(d_data, filas, columnas);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

Matriz2D Matriz2D::MultiplicarCUDA(const Matriz2D& B) const {
    if (columnas != B.filas)
        throw std::invalid_argument("Dimensiones incompatibles para multiplicación");

    Matriz2D R(filas, B.columnas);

    float *d_A, *d_B, *d_C;
    size_t sizeA = filas * columnas * sizeof(float);
    size_t sizeB = B.filas * B.columnas * sizeof(float);
    size_t sizeC = filas * B.columnas * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, datos, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.datos, sizeB, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((B.columnas + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (filas + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, filas, columnas, B.columnas);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(R.datos, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return R;
}

void Matriz2D::NormalizarFilas_CUDA() {
    size_t size = filas * columnas;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, datos, size * sizeof(float), cudaMemcpyHostToDevice));

    size_t shared_mem = 2 * sizeof(float);
    normalizeKernel<<<filas, columnas, shared_mem>>>(d_data, filas, columnas);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(datos, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}
