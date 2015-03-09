#include <cuda.h>
#include "CuMatrixUtils.h"
#include "CuSync.h"
#include "CuMatrixProcedures/CuCompareProcedures.h"

extern "C" __global__ void CUDAKernel_Test1(int* output, int* buffer1,
        size_t offset, size_t length,
        int* bin, int* bout) {
    /*
    cuda_debug_function();
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    cuda_debug_function();
    uintt tindex = threadIndexX + offset * threadIndexY;
    buffer1[tindex] = 1;
    __syncblocks(bin, bout);
    cuda_debug("output1 = %d", buffer1[0]);
    do {
        cuda_compare_step_2(buffer1);
        __syncblocks(bin, bout);
        cuda_debug("output2 = %d", buffer1[0]);
    } while (length > 1);
     *output = buffer1[0];
    cuda_debug("output3 = %d", buffer1[0]);
    cuda_debug_function();*/
}

extern "C" __global__ void CUDAKernel_Test2(int* output, int* buffer1,
        size_t offset, size_t length,
        int* bin) {
    /*
    cuda_debug_function();
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    cuda_debug_function();
    uintt tindex = threadIndexX + offset * threadIndexY;
    buffer1[tindex] = 1;
    __syncblocksAtomic(bin);
    cuda_debug("output1 = %d", buffer1[0]);
    do {
        cuda_compare_step_2(buffer1);
        __syncblocksAtomic(bin);
        cuda_debug("output2 = %d", buffer1[0]);
    } while (length > 1);
     *output = buffer1[0];
    cuda_debug("output3 = %d", buffer1[0]);
    cuda_debug_function();*/
}
