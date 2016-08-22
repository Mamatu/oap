/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef CU_SYNC_H
#define	CU_SYNC_H

#include <stdio.h>
#include "CuCore.h"
#include "DebugLogs.h"

#define DEBUG

/*
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
*/

#ifdef CUDA

#ifdef DEBUG
#define cuda_lock() __threadfence(); __syncthreads();
#else
#define cuda_lock() __syncthreads();
#endif

#else
#define cuda_lock() 
#endif


#endif
