#include "Dim3.h"

#ifdef CUDATEST

Dim3 blockIdx;
Dim3 blockDim;
Dim3 gridDim;

void ResetCudaCtx() {
    blockIdx.clear();
    blockDim.clear();
    gridDim.clear();
}

ThreadIdx::ThreadIdxs ThreadIdx::m_threadIdxs;

void ThreadIdx::clear() {
    threadIdx.clear();
}

#endif