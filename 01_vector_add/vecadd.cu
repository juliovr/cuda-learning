#include <stdio.h>
#include <stdlib.h>


#define cudaMalloc(dev_ptr, size)                                                                           \
    err = cudaMalloc(dev_ptr, size);                                                                        \
    if (err != cudaSuccess) {                                                                               \
        fprintf(stderr, "Failed to allocate memory (line %d): %s\n", __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                                                                 \
    }

#define cudaMemcpy(dst, src, count, kind)                                                                   \
    err = cudaMemcpy(dst, src, count, kind);                                                                \
    if (err != cudaSuccess) {                                                                               \
        fprintf(stderr, "Failed to copy memory (line %d): %s\n", __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                                                                 \
    }

#define cudaFree(dev_ptr)                                                                                   \
    err = cudaFree(dev_ptr);                                                                                \
    if (err != cudaSuccess) {                                                                               \
        fprintf(stderr, "Failed to free memory (line %d): %s\n", __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                                                                 \
    }


__global__
void vecadd_kernel(float *A, float *B, float *C, int n)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecadd(float *A, float *B, float *C, int n)
{
    cudaError_t err = cudaSuccess;

    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Alloc memory in the device (GPU)
    printf("Allocating memory on the GPU\n");
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // Copy the data from the host to the device in the previously allocated memory
    printf("Copy data host -> device\n");
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Call the kernel
    printf("Call the kernel\n");
    vecadd_kernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Copy the result of the procedure back to the host
    printf("Copy data device -> host\n");
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // Free memory allocated on the GPU
    printf("Free memory allocated on the GPU\n");
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    float A[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, };
    float B[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, };
    int n = sizeof(A) / sizeof(float);

    //    float *C = (float *)malloc(n * sizeof(float));
    float C[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, };
    vecadd(A, B, C, n);

    for (int i = 0; i < n; i++) {
        printf("%.2f ", C[i]);
    }

    printf("\n");

    return 0;
}
