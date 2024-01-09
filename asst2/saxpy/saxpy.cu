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


__global__ void
gemm_kernel(int N, int M, int K, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sABlock[N][K];
    __shared__ float sBBlock[K][M];
    __shared__ float sCBlock[N][M];

    
}

#define OUT
__device__ void
block_gemm(int blockKDim, float** sABlock, float** sBBlock, OUT float** sCBlock, int threadYDim, int threadXDim){
    float frag_a[threadYDim], frag_b[threadXDim];
    float acc[threadYDim][threadXDim];
    
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
