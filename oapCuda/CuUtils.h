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



#ifndef CU_UTILS_H
#define	CU_UTILS_H

#include <cuda.h>
#include <stdio.h>
#include "Math.h"
#include "CuCore.h"
#include "CuSync.h"

#ifdef CUDA

#ifdef DEBUG
#define CUDA_DEBUG() \
{ \
    const uintt tx = blockIdx.x * blockDim.x + threadIdx.x;\
    const uintt ty = blockIdx.y * blockDim.y + threadIdx.y;\
    if (tx == 0 && ty == 0) { \
        printf("%s %s %d \n", __FUNCTION__, __FILE__, __LINE__);\
    }\
}
#else
#define CUDA_DEBUG()
#endif


#ifndef DEBUG
#define cuda_debug_buffer(s, buffer, len)
#define cuda_debug_matrix_ex(s, mo)
#define cuda_debug(x, ...)
#define cuda_debug_function()
#define cuda_debug_thread(tx, ty, arg, ...)
#else
#define cuda_debug_matrix_ex(s, mo) \
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s = \n",s);\
        CUDA_PrintMatrixEx(mo);\
    }\
}

#define cuda_debug_buffer(s, buffer, len) \
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s = \n",s);\
        CUDA_PrintBuffer(buffer, len);\
    }\
}

#define cuda_debug_buffer_uint(s, buffer, len) \
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s = \n",s);\
        CUDA_PrintBufferUintt(buffer, len);\
    }\
}

#define cuda_debug_buffer_int(s, buffer, len) \
{\
        printf("%s = \n",s);\
        CUDA_PrintBufferInt(buffer, len);\
}

#define cuda_debug(arg, ...)\
{\
    if ((threadIdx.x)==0\
        && (threadIdx.y)==0) {\
        printf(arg, ##__VA_ARGS__);\
        printf("\n");\
    }\
}

#define cuda_debug_abs(arg, ...)\
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf(arg, ##__VA_ARGS__);\
        printf("\n");\
    }\
}

#define cuda_debug_function()\
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s %s %d \n", __FUNCTION__,__FILE__,__LINE__);\
    }\
}

#define cuda_debug_thread(tx, ty, arg, ...)\
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==tx\
        && (blockIdx.y * blockDim.y + threadIdx.y)==ty) {\
        printf("%s %s %d Thread: %u %u: ", __FUNCTION__,__FILE__,__LINE__, tx, ty);\
        printf(arg, ##__VA_ARGS__);\
    }\
}

#endif


extern "C" __device__ void CUDA_PrintBuffer(floatt* buffer, uintt length) {
    for (uintt fa = 0; fa < length; ++fa) {
        printf("buffer[%u] = %f \n", fa, buffer[fa]);
    }
}

extern "C" __device__ void CUDA_PrintBufferUintt(uintt* buffer, uintt length) {
    for (uintt fa = 0; fa < length; ++fa) {
        printf("buffer[%u] = %llu \n", fa, buffer[fa]);
    }
}

extern "C" __device__ void CUDA_PrintBufferInt(int* buffer, uintt length) {
    for (uintt fa = 0; fa < length; ++fa) {
        printf("buffer[%u] = %d \n", fa, buffer[fa]);
    }
}

extern "C" __device__ void CUDA_PrintFloat(floatt v) {
    printf("[%f]", v);
    printf("\n");
}

extern "C" __device__ void CUDA_PrintInt(uintt v) {
    printf("[%u]", v);
    printf("\n");
}


#endif

#endif	/* CUUTILS_H */
