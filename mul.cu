#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <curand_kernel.h>
#include <omp.h>

#define TILE_SIZE 32

__global__ void generate_random(float *array, int n, unsigned seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        array[idx] = curand_uniform(&state);
    }
}
void cpu_mult(float *a, float *b, float *c, size_t m, size_t n, size_t p)
{
#pragma omp paraller
    {
        if (omp_get_thread_num() == 0)
        {
            printf("Using %d threads for OpenMp\n", omp_get_num_threads());
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                c[i * p + j] = 0;
                for (int k = 0; k < n; k++)
                {
                    c[i * p + j] = a[i * n + k] * b[k * p + j];
                }
            }
        }
    }
}
__global__ void mult_gpu_naive(float *a, float *b, float *c, size_t m, size_t n, size_t p)
{
    unsigned int crow = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int cul = threadIdx.y + blockDim.y * blockIdx.y;
    if (crow < m && cul < p)
    {
        float temp = 0;
        for (int k = 0; k < n; k++)
        {
            temp += a[crow * n + k] * b[k * p + cul];
        }
        c[crow * p + cul] = temp;
    }
}
__global__ void mult_gpu_tiled(float *a, float *b, float *c, size_t m, size_t n, size_t p)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row =blockIdx.y*TILE_SIZE+threadIdx.y;
    int col=blockIdx.x*TILE_SIZE+threadIdx.x;
    float sum=0.0f;
    for(int t=0;t<(n+TILE_SIZE-1)/TILE_SIZE;t++)
    {
        if(row<m && (t*TILE_SIZE+threadIdx.x)<n)
            tileA[threadIdx.y][threadIdx.x]=a[row*n+t*TILE_SIZE+threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x]=0.0f;
        if(col<p && (t*TILE_SIZE+threadIdx.y)<n)
            tileB[threadIdx.y][threadIdx.x]=b[(t*TILE_SIZE+threadIdx.y)*p+col];
        else 
            tileB[threadIdx.y][threadIdx.x]=0.0f;
        __syncthreads();

        for(int k=0;k<TILE_SIZE;k++)
        {
            sum +=tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row<m && col<p)
    {
        c[row*p + col]=sum;
    }

}
int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "need 4 args \n");
        exit(-1);
    }
    size_t A1 = strtoul(argv[1], NULL, 0);
    size_t A2 = strtoul(argv[2], NULL, 0);
    size_t B1 = strtoul(argv[3], NULL, 0);
    size_t B2 = strtoul(argv[4], NULL, 0);
    if (A2 != B1)
    {
        fprintf(stderr, "wrong input \n");
        exit(-1);
    }
    size_t A = A1 * A2;
    size_t B = B1 * B2;
    size_t C = A1 * B2;
    float *d_A, *d_B, *d_C, *h_A, *h_B, *h_C;
    cudaMalloc((float **)&d_A, A * sizeof(float));
    cudaMalloc((float **)&d_B, B * sizeof(float));
    cudaMalloc((float **)&d_C, C * sizeof(float));
    dim3 block_A(128);
    dim3 block_B(128);
    dim3 grid_A((A + block_A.x - 1) / block_A.x);
    dim3 grid_B((B + block_B.x - 1) / block_B.x);
    generate_random<<<grid_A, block_A>>>(d_A, A, 42);
    generate_random<<<grid_B, block_B>>>(d_B, B, 45);
    h_A = (float *)malloc(A * sizeof(float));
    h_B = (float *)malloc(B * sizeof(float));
    h_C = (float *)malloc(C * sizeof(float));
    
    cudaMemcpy(h_A, d_A, A * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, B * sizeof(float), cudaMemcpyDeviceToHost);
    

    
    dim3 block(32, 32);
    dim3 grid((A1 + block.x - 1) / block.x, (B2 + block.y - 1) / block.y);
    mult_gpu_naive<<<grid,block>>>(d_A, d_B, d_C, A1, A2, B2);
    cpu_mult(h_A, h_B, h_C, A1, A2, B2);
    cudaDeviceSynchronize();
    free(h_C);
    free(h_B);
    free(h_A);
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    cudaDeviceReset();
    return 0;
}
