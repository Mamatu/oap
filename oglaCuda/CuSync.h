/* 
 * File:   CuUtils.h
 * Author: mmatula
 *
 * Created on October 26, 2014, 2:33 PM
 */

#ifndef CU_SYNC_H
#define	CU_SYNC_H

#include <cuda.h>
#include <stdio.h>

__device__ int aux;

extern "C" __device__ void __syncblocks(int* syncBufferIn, int* syncBufferOut) {
    __threadfence();
    aux = 0;
    //float aux = 0;
    syncBufferIn[blockIdx.x + gridDim.x * blockIdx.y] = 1;
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        while (syncBufferIn[threadIdx.x + blockDim.x * threadIdx.y] != 1) {
            aux++;
            __syncthreads();
        }
        syncBufferOut[threadIdx.x + blockDim.x * threadIdx.y] = 1;
    }
    while (syncBufferOut[blockIdx.x + gridDim.x * blockIdx.y] != 1) {
        aux++;
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        syncBufferIn[blockIdx.x + gridDim.x * blockIdx.y] = 0;
        syncBufferOut[blockIdx.x + gridDim.x * blockIdx.y] = 0;
    }
}

extern "C" __device__ void __syncblocksAtomic(int* mutex) {
    __threadfence();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(mutex, 1);
        while (*mutex != gridDim.x * gridDim.y) {
            __syncthreads();
        }
    }
}

#endif	

