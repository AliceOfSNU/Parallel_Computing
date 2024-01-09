#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

//kernel

//shmem impl defines
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define BLOCKSIZE 512
#define SCAN_BLOCK_DIM   BLOCKSIZE
#define OUT

// code from NVIDIA
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// assuming size <= WARP_SIZE
inline __device__ int
warp_inclusive_scan(int threadIndex, int idata, volatile int *s_Data, int size){
    // should know this trick by now..
    int pos = 2 * threadIndex - (threadIndex & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];

    return s_Data[pos];
}

//assuming size <= WARP_SIZE
inline __device__ int warp_exclusive_scan(int threadIndex, int idata, volatile int *sScratch, int size){
    // Inclusive() - myself(idata)
    return warp_inclusive_scan(threadIndex, idata, sScratch, size) - idata;
}

//implementation using above(given) code
inline __device__ int
exclusive_scan_shmem(
    int threadIndex,
    int* sInput,    //on shared memory
    OUT int* sOutput,   //on shared memory
    volatile int* sScratch, //on shared memory
    int size
){
    //no divergence here..
    if(size > WARP_SIZE){
        int idata = sInput[threadIndex];
        //calculate prefix sum within the WARP block. (mod 32)
        //that is, 
        // WARP 1
        //  res[32] = 32:32
        //  res[33] = 32:33
        //  ... 
        //  res[63] = 32:63
        // WARP 2
        //  res[64] = 65:64
        //  res[65] = 64:65
        //  ...
        //  res[95] = 64:95

        int warpResult = warp_inclusive_scan(threadIndex, idata, sScratch, WARP_SIZE);

        __syncthreads();

        // save each WARP's exclusive sum compactly to sScratch
        // sScratch[0] = 0 (starts with all elements 0)
        // sScratch[1] = 0:31
        // sScratch[2] = 32:63
        // ...
        // sScratch[NUM_WARPS] = 32*(NUM_WARPS-1):32*NUM_WARPS
        if ((threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1)) 
            sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;
        

        __syncthreads();


        //run prefix sum on above array
        //if BLOCK has 512 threads, there are 512/32 = 16 warps
        //for warp0: 0
        //  warp[1]: 0:31
        //  warp[2]: 0:63
        //  warp[3]: 0:95...
        if(threadIndex < (SCAN_BLOCK_DIM/WARP_SIZE)){
            int val = sScratch[threadIndex];
            sScratch[threadIndex] = warp_exclusive_scan(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
        }

        __syncthreads();

        // add 'warp-prefix' + 'within_warp prefix' - me
        sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE] - idata;
    
    } else if (threadIndex < WARP_SIZE) {
        //threads 0, ... , 31
        int idata = sInput[threadIndex];
        sOutput[threadIndex] = warp_exclusive_scan(threadIndex, idata, sScratch, size);
    }
}

__global__ void
exclusive_scan_optimized_kernel(int* device_data, OUT int* device_out, int length){
    //each thread in the block works on one output,
    //this is OUT_OF_PLACE operation which fills output..
    __shared__ int prefixSumInput[BLOCKSIZE];
    __shared__ int prefixSumOutput[BLOCKSIZE];
    __shared__ int prefixSumScratch[2 * BLOCKSIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // copy data onto input
    if(tid < length) prefixSumInput[tid] = device_data[tid];

    // run algo.
    exclusive_scan_shmem(tid, prefixSumInput, OUT prefixSumOutput, prefixSumScratch, length);
    
    // copy result into device_out
    if(tid < length) OUT device_out[tid] = prefixSumOutput[tid];

}


//length can be anything < threadsPerBlock(512 for now)
void exclusive_scan_optimized(int* device_data, int length)
{
    const int threadsPerBlock = nextPow2(length);
    exclusive_scan_optimized_kernel<<<1, threadsPerBlock>>>(device_data, OUT device_data, threadsPerBlock);
}

// my implementation
__global__ void 
down_sweep_kernel(int* arr, int length, int level){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length && ((idx + 1) % (level << 1)) == 0)
        arr[idx] += arr[idx - level];
}

__global__ void
up_sweep_kernel(int* arr, int length, int level){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(level == -1) {
        if (idx == length - 1) arr[idx] = 0; //this needs to be done once.
        return;
    }
    if(idx < length && ((idx + 1) % (level << 1) == 0)) {
        int swap = arr[idx - level];
        arr[idx - level] = arr[idx];
        arr[idx] += swap;
    }

}

void exclusive_scan(int* device_data, int length)
{
    /* 
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
    int rounded_length = nextPow2(length);

    const int threadsPerBlock = 512;
    const int blocks = (rounded_length + threadsPerBlock - 1) / threadsPerBlock;
    // sweep
    for (int level = 1; level <= (rounded_length >> 2); level <<= 1){
        down_sweep_kernel<<<blocks, threadsPerBlock>>>(device_data, rounded_length, level);
    }
    // set last 0
    up_sweep_kernel<<<blocks, threadsPerBlock>>>(device_data, rounded_length, -1);
    // gather
    for( int level = (rounded_length >> 1); level > 0; level >>= 1){
        up_sweep_kernel<<<blocks, threadsPerBlock>>>(device_data, rounded_length, level);
    }

}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    //exclusive_scan(device_data, end - inarray);
    exclusive_scan_optimized(device_data, end - inarray);
    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}



int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    return 0;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
