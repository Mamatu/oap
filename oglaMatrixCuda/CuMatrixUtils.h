/* 
 * File:   CuMatrixUtils.h
 * Author: mmatula
 *
 * Created on October 26, 2014, 2:34 PM
 */

#ifndef OGLA_CU_MATRIXUTILS_H
#define	OGLA_CU_MATRIXUTILS_H

#include "CuUtils.h"
#include "Matrix.h"
#include "Buffer.h"
#include "MatrixEx.h"

#ifndef DEBUG
#define cuda_debug_matrix(s, mo)
#define cuda_debug_buffer(s, buffer, len)
#define cuda_debug_matrix_ex(s, mo)
#define cuda_debug(x, ...)
#define cuda_debug_function()
#define cuda_debug_thread(tx, ty, arg, ...)
#else
#define cuda_debug_matrix(s, mo) \
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s = \n",s);\
        CUDA_PrintMatrix(mo);\
    }\
}

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

#define cuda_debug(arg, ...)\
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

extern "C" __device__ void CUDA_PrintMatrix(math::Matrix* m) {
    for (uintt fb = 0; fb < m->rows; ++fb) {
        printf("[");
        for (uintt fa = 0; fa < m->columns; ++fa) {
            printf("(%f", m->reValues[fb * m->columns + fa]);
            if (m->imValues) {
                printf(",%f", m->imValues[fb * m->columns + fa]);
            }
            printf(")");
        }
        printf("]");
        printf("\n");
    }
}

extern "C" __device__ void CUDA_PrintMatrixEx(const MatrixEx& m) {
    printf("columns: %u %u \n", m.bcolumn, m.ecolumn);
    printf("rows: %u %u \n", m.brow, m.erow);
    printf("offset: %u %u \n", m.boffset, m.eoffset);
}

extern "C" __device__ void CUDA_PrintBuffer(floatt* buffer, uintt length) {
    for (uintt fa = 0; fa < length; ++fa) {
        printf("buffer[%u] = %f \n", fa, buffer[fa]);
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

#endif	/* CUMATRIXUTILS_H */

