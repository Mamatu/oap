/* 
 * File:   CuMatrixUtils.h
 * Author: mmatula
 *
 * Created on October 26, 2014, 2:34 PM
 */

#ifndef OGLA_CU_MATRIXUTILS_H
#define	OGLA_CU_MATRIXUTILS_H

#include "CuUtils.h"

#ifndef DEBUG
#define cuda_debug_structure(s, mo)
#define cuda_debug_matrix(s, mo)
#define cuda_debug(x, ...)
#define cuda_debug_function()
#else
#define cuda_debug_structure(s, mo) \
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s = \n",s);\
        CUDA_PrintMatrixS(mo);\
    }\
}

#define cuda_debug_matrix(s, mo) \
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s = \n",s);\
        CUDA_PrintMatrix(mo);\
    }\
}

#define cuda_debug(arg, ...)\
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf(arg, ##__VA_ARGS__);\
    }\
}

#define cuda_debug_function()\
{\
    if ((blockIdx.x * blockDim.x + threadIdx.x)==0\
        && (blockIdx.y * blockDim.y + threadIdx.y)==0) {\
        printf("%s %s %d \n", __FUNCTION__,__FILE__,__LINE__);\
    }\
}

#endif

extern "C" __device__ void CUDA_PrintMatrixS(MatrixStructure* ms) {
    for (uintt fb = ms->m_beginRow;
            fb < ms->m_beginRow + ms->m_subrows; ++fb) {
        printf("[");
        for (uintt fa = ms->m_beginColumn;
                fa < ms->m_beginColumn + ms->m_subcolumns; ++fa) {
            printf("(%f", ms->m_matrix->reValues[ms->m_beginColumn + fb * ms->m_subcolumns + fa]);
            if (ms->m_matrix->imValues) {
                printf(",%f", ms->m_matrix->imValues[ms->m_beginColumn + fb * ms->m_subcolumns + fa]);
            }
            printf(")");
        }
        printf("]");
        printf("\n");
    }
}

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

extern "C" __device__ void CUDA_PrintFloat(floatt v) {
    printf("[%f]", v);
    printf("\n");
}

extern "C" __device__ void CUDA_PrintInt(uintt v) {
    printf("[%u]", v);
    printf("\n");
}

#endif	/* CUMATRIXUTILS_H */

