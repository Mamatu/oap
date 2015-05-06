/*
 * File:   CuMatrixUtils.h
 * Author: mmatula
 *
 * Created on October 26, 2014, 2:34 PM
 */

#ifndef OGLA_CU_MATRIXUTILS_H
#define	OGLA_CU_MATRIXUTILS_H

#include "CuUtils.h"
#include "CuCore.h"
#include "Matrix.h"
#include "Buffer.h"
#include "MatrixEx.h"

#ifndef CUDATEST

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

#ifndef DEBUG
#define cuda_debug_matrix(s, mo)
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

#endif

#endif

#endif	/* CUMATRIXUTILS_H */

