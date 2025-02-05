#include <stdio.h>

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
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Alloc memory in the device (GPU)
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // Copy the data from the host to the device in the previously allocated memory
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Call the kernel
    vecadd_kernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    float A[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, };
    float B[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, };
    int n = sizeof(A) / sizeof(float);
    printf("n = %d\n", n);

    float C[n];
    vecadd(A, B, C, n);

    for (int i = 0; i < n; i++) {
        printf("%.2f ", C[i]);
    }

    printf("\n");

    return 0;
}
