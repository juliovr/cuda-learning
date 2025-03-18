#include <stdio.h>

#include "../util.h"

#define MATRIX_N 3

__global__
void matrix_multiply_kernel(float *c, float *a, float *b, int width, int height)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        float sum = 0;
        for (int i = 0; i < width; ++i) {
            sum += a[row*width + i]*b[i*width + col];
        }

        c[row*width + col] = sum;
    }
}

void matrix_multiply_cpu(float *c, float *a, float *b, int width, int height)
{
    for (int i = 0; i < width; ++i) {
        for (int row = 0; row < height; ++row) {
            float sum = 0;
            for (int col = 0; col < width; ++col) {
                sum += a[row*width + col]*b[col*width + i];
            }

            c[row*width + i] = sum;
        }
    }
}

void print_matrix(float *matrix, int width, int height)
{
    for (int row = 0; row < height; ++row) {
        printf("|");
        for (int col = 0; col < width; ++col) {
            printf(" %.2f ", matrix[row*width + col]);
        }
        printf("|\n");
    }
}

/*
| 30.00  24.00  18.00 |
| 84.00  69.00  54.00 |
| 138.00  114.00  90.00 |
*/
int main()
{
    float a[MATRIX_N][MATRIX_N] = {
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f },
        { 7.0f, 8.0f, 9.0f },
    };

    float b[MATRIX_N][MATRIX_N] = {
        { 9.0f, 8.0f, 7.0f },
        { 6.0f, 5.0f, 4.0f },
        { 3.0f, 2.0f, 1.0f },
    };

    float c[MATRIX_N][MATRIX_N];

#if 1
    cudaError_t err = cudaSuccess;

    int matrix_size = MATRIX_N*MATRIX_N*sizeof(float);

    float *a_d;
    float *b_d;
    float *c_d;
    cudaMalloc((void **)&a_d, matrix_size);
    cudaMalloc((void **)&b_d, matrix_size);
    cudaMalloc((void **)&c_d, matrix_size);

    cudaMemcpy(a_d, a, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, matrix_size, cudaMemcpyHostToDevice);

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(3, 3, 1);
    matrix_multiply_kernel<<<dim_grid, dim_block>>>(c_d, a_d, b_d, MATRIX_N, MATRIX_N);

    cudaMemcpy(c, c_d, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(c_d);
    cudaFree(b_d);
    cudaFree(a_d);
#else
    matrix_multiply_cpu((float *)c, (float *)a, (float *)b, MATRIX_N, MATRIX_N);
#endif
    
    print_matrix((float *)c, 3, 3);

    return 0;
}