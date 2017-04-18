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




#ifndef OAP_CU_MATRIXUTILS_H
#define	OAP_CU_MATRIXUTILS_H

#include "CuUtils.h"
#include "CuCore.h"
#include "Matrix.h"
#include "MatrixEx.h"

#ifdef CUDA

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
    printf("columns: %u %u \n", m.beginColumn, m.columnsLength);
    printf("rows: %u %u \n", m.beginRow, m.rowsLength);
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
