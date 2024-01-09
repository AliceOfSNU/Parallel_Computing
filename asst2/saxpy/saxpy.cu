#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    // allocate device memory buffers on the GPU using cudaMalloc
    size_t size = N * sizeof(float);
    cudaMalloc(&device_x, size);
    cudaMalloc(&device_y, size);
    cudaMalloc(&device_result, size);

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_x, xarray, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, size, cudaMemcpyHostToDevice);

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaThreadSynchronize(); //must synchronize before timing
    double kernelEndTime = CycleTimer::currentSeconds();
    // TODO copy result from GPU using cudaMemcpy
    cudaMemcpy(resultarray, device_result, size, cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("Kernel: %.3f ms\n", 1000.f*kernelDuration);

    // free memory buffers on the GPU
    cudaFree(device_x); cudaFree(device_y); cudaFree(device_result);
}

// MM kernels
#define OUT
#define CEIL_DIV(x, y) (((x)+(y)-1)/(y))
#define BLOCK_SIZE 1024

__global__ void
gemm_kernel_naive(int M, int N, int K, 
    const float* A, 
    const float* B, 
    OUT float* C
) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    //int tid = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);

    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(tidx < M && tidy < N){
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[tidx * K + i] * B[i * N + tidy];
        }
        C[tidx * N + tidy] = tmp;
    }
    
}

__global__ void
gemm_kernel_coalesced(int M, int N, int K, 
    const float* A, 
    const float* B, 
    OUT float* C
) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    //int tid = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);

    const uint tidx = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    const uint tidy = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    
    if(tidy < M && tidx < N){
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[tidy * K + i] * B[i * N + tidx];
        }
        C[tidy * N + tidx] = tmp;
    }
    
}



// hierarical GEMM
// algorithm adapted from CuTe, Cutlass

// this code runs in a single warp!
// a warp always calculates outer product of 8*1 frag A and 1*4 frag B
// and fills 8*4 block of C.
// if tiling to larger spans, run outer loop which tiles into 8*4 sections, 
// don't need syncs.
__device__ void warp_gemm(int tid, int fragY, int fragX, float** sA, float** sB, OUT float** sC){
    //fill in the answers~
    int r = tid >> 0x08, c = tid & 0x08;
    sC[r][c] += sA[r][0] * sB[0][c];
}

__device__ void
block_gemm(int blockKDim, float** sABlock, float** sBBlock, OUT float** sCBlock, 
    int threadYDim, int threadXDim)
{
    float frag_a[64*64], frag_b[64*64];
    float acc[64*64];
    
    int blockX = 8, blockY = 4, blockTileK = 8;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid % blockDim.x == 0){
        //one per block. copy to shared mem
    }
    __syncthreads();//wait.

    int warpXInBlock = 4, warpYInBlock = 2;
    int warpXIdx = (tid >> 5) % warpXInBlock, warpIdxY = (tid >> 5 + warpXInBlock - 1)/warpXInBlock;
    
    #pragma unroll
    for(int fragK = 0; fragK < blockTileK ; fragK++){
        #pragma unroll
        for(int fragYIdx = 0; fragYIdx < warpYInBlock; ++fragYIdx){
            #pragma unroll
            for(int fragXIdx = 0; fragXIdx < warpXInBlock; ++fragXIdx){
                warp_gemm(tid, fragYIdx, fragXIdx, sABlock, sBBlock, sCBlock);
            }
        }
    }

}



// calling code.
# define ALGO ALGO_NAIVE
# define ALGO_NAIVE 1
# define ALGO_COAL 2
void
sgemm(int M, int N, int K, 
    const float* A,
    const float* B,
    OUT float* C)
{
    // allocate memory on device.
    float* device_A;
    float* device_B;
    float* device_C;
    cudaMalloc(&device_A, M * K * sizeof(float));
    cudaMalloc(&device_B, K * N * sizeof(float));
    cudaMalloc(&device_C, M * N * sizeof(float));

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("%dx%d @ %dx%d multiplication\n", M, K, K, N);
    // run kernel with timer


    double kernelStartTime = CycleTimer::currentSeconds();

    ////// IMPLEMENTATION HERE ///////
#if ALGO == ALGO_NAIVE
    //naive 39ms on 256*16 @ 16*256 task
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);
    gemm_kernel_naive<<<gridDim, blockDim>>>(M, N, K, device_A, device_B, device_C);
#elif ALGO == ALGO_COAL
    //coalesced 26ms on 256*16 @ 16*256 task
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(BLOCK_SIZE); //1dim threads
    gemm_kernel_coalesced<<<gridDim, blockDim>>>(M, N, K, device_A, device_B, OUT device_C);
#endif
    ////// END IMPL //////

    cudaThreadSynchronize(); //must synchronize before timing

    double kernelEndTime = CycleTimer::currentSeconds();
    // copy result from GPU using cudaMemcpy
    cudaMemcpy(C, device_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\n", 1000.f * overallDuration);

    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("Kernel: %.3f ms\n", 1000.f*kernelDuration);

    // free memory buffers on the GPU
    cudaFree(device_A); cudaFree(device_B); cudaFree(device_C);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
