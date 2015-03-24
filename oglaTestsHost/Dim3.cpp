#include "Dim3.h"

#ifdef CUDATEST

Dim3 threadIdx;
Dim3 blockIdx;
Dim3 blockDim;
Dim3 gridDim;

void ResetCudaCtx() {
    threadIdx.clear();
    blockIdx.clear();
    blockDim.clear();
    gridDim.clear();
}

#endif