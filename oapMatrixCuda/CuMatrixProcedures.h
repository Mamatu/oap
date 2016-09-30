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




#ifndef OAP_CU_MATRIXPROCEDURES_H
#define	OAP_CU_MATRIXPROCEDURES_H

#include <cuda.h>
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"
#include "CuCore.h"

#include "CuMatrixProcedures/CuCompareProcedures.h"
#include "CuMatrixProcedures/CuCompareOptProcedures.h"
#include "CuMatrixProcedures/CuCompareOptProcedures2.h"
#include "CuMatrixProcedures/CuCopyProcedures.h"
#include "CuMatrixProcedures/CuInversionProcedures.h"
#include "CuMatrixProcedures/CuDotProductProcedures.h"
#include "CuMatrixProcedures/CuDotProductOptProcedures.h"
#include "CuMatrixProcedures/CuMultiplicationProcedures.h"
#include "CuMatrixProcedures/CuAdditionProcedures.h"
#include "CuMatrixProcedures/CuSubstractionProcedures.h"
#include "CuMatrixProcedures/CuTransposeProcedures.h"
#include "CuMatrixProcedures/CuIdentityProcedures.h"
#include "CuMatrixProcedures/CuQRProcedures.h"
#include "CuMatrixProcedures/CuIsUpperTriangularProcedures.h"
#include "CuMatrixProcedures/CuMagnitudeOptProcedures.h"
#include "CuMatrixProcedures/CuMagnitudeOptProcedures2.h"
#include "CuMatrixProcedures/CuTriangularH.h"

__hostdevice__ void CUDA_setDiagonalReMatrix(
    math::Matrix* dst,
    floatt v,
    uintt threadIndexX,
    uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->reValues[index] = v;
    } else {
        dst->reValues[index] = 0;
    }
    threads_sync();
}

__hostdevice__ void CUDA_setDiagonalImMatrix(
    math::Matrix* dst,
    floatt v,
    uintt threadIndexX,
    uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->imValues[index] = v;
    } else {
        dst->imValues[index] = 0;
    }
    threads_sync();
}

__hostdevice__ void CUDA_setDiagonalMatrix(
    math::Matrix* dst,
    floatt rev,
    floatt imv,
    uintt threadIndexX,
    uintt threadIndexY) {
    if (NULL != dst->reValues) {
        CUDA_setDiagonalReMatrix(dst, rev, threadIndexX, threadIndexY);
    } else if (NULL != dst->imValues) {
        CUDA_setDiagonalImMatrix(dst, imv, threadIndexX, threadIndexY);
    }
}

__hostdeviceinline__ void CUDA_tensorProductReMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    const intt bcolumn = threadIndexX;
    const intt brow = threadIndexY;
    const intt columns = output->columns;
    const intt columns1 = params0->columns;
    const intt columns2 = params1->columns;
    const intt c1 = params0->columns;
    const intt c2 = params1->columns;
    intt fa = bcolumn;
    intt fb = brow;
    intt fa1 = fa / c1;
    intt fa2 = fa % c2;
    intt fb1 = fb / c1;
    intt fb2 = fb % c2;
    intt index2 = (fa + columns * fb);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->reValues[index2] =
        params0->reValues[index] *
        params1->reValues[index1];
    threads_sync();
}

__hostdeviceinline__ void CUDA_tensorProductImMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    const intt bcolumn = threadIndexX;
    const intt brow = threadIndexY;
    const intt columns = output->columns;
    const intt columns1 = params0->columns;
    const intt columns2 = params1->columns;
    const intt c1 = params0->columns;
    const intt c2 = params1->columns;
    intt fa = bcolumn;
    intt fb = brow;
    intt fa1 = fa / c1;
    intt fa2 = fa % c2;
    intt fb1 = fb / c1;
    intt fb2 = fb % c2;
    intt index2 = (fa + columns * fb);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->reValues[index2] =
        -params0->imValues[index] *
        params1->imValues[index1];
    threads_sync();
}

__hostdeviceinline__ void CUDA_tensorProductMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    const uintt columns = output->columns;
    const uintt columns1 = params0->columns;
    const uintt columns2 = params1->columns;
    const uintt c1 = params0->columns;
    const uintt c2 = params1->columns;
    intt fa1 = threadIndexX / c1;
    intt fa2 = threadIndexX % c2;
    intt fb1 = threadIndexY / c1;
    intt fb2 = threadIndexY % c2;
    intt index2 = (threadIndexX + columns * threadIndexY);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->reValues[index2] =
        params0->reValues[index] *
        params1->reValues[index1] -
        params0->imValues[index] *
        params1->imValues[index1];
    output->imValues[index2] =
        params0->reValues[index] *
        params1->imValues[index1] -
        params0->imValues[index] *
        params1->reValues[index1];
    threads_sync();
}

__hostdeviceinline__ void conjugateIm(math::Matrix* output,
    math::Matrix * params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    output->imValues[index] = -params0->imValues[index];
}

__hostdeviceinline__ void conjugateIm1(math::Matrix * output,
    uintt tx, uintt ty) {
    uintt index = tx + output->columns * ty;
    output->imValues[index] = -output->imValues[index];
}

__hostdevice__ void CUDA_setSubRows(math::Matrix* matrix,
    uintt brow, uintt erow) {
    matrix->rows = erow;
}

__hostdevice__ void CUDA_setSubColumns(math::Matrix* matrix,
    uintt bcolumn, uintt ecolumn) {
    matrix->columns = ecolumn;
}

__hostdevice__ void CUDA_setVector(math::Matrix* V, uintt column,
    math::Matrix* v, uintt length, uintt tx, uintt ty) {
    if (ty < length) {
        uintt index1 = ty * V->columns + column + tx;
        uintt index2 = ty * v->columns + tx;
        if (V->reValues != NULL && v->reValues != NULL) {
            V->reValues[index1] = v->reValues[index2];
        }
        if (V->imValues != NULL && v->imValues != NULL) {
            V->imValues[index1] = v->imValues[index2];
        }
    }
    threads_sync();
}

__hostdevice__ void CUDA_getVector(math::Matrix* v, uintt length,
    math::Matrix* V, uintt column, uintt tx, uintt ty) {
    if (ty < length) {
        uintt index1 = ty * V->columns + column + tx;
        uintt index2 = ty * v->columns + tx;
        if (V->reValues != NULL && v->reValues != NULL) {
            v->reValues[index2] = V->reValues[index1];
        }
        if (V->imValues != NULL && v->imValues != NULL) {
            v->imValues[index2] = V->imValues[index1];
        }
    }
    threads_sync();
}

__hostdevice__ void CUDA_setZeroMatrix(math::Matrix* matrix,
    uintt tx, uintt ty) {
    if (tx < matrix->columns && ty < matrix->rows) {
        if (NULL != matrix->reValues) {
            matrix->reValues[ty * matrix->columns + tx] = 0;
        }
        if (NULL != matrix->imValues) {
            matrix->imValues[ty * matrix->columns + tx] = 0;
        }
    }
}

__hostdevice__ void CUDA_setIdentityMatrix(math::Matrix* matrix,
    uintt tx, uintt ty) {
    floatt v = 0;
    if (tx == ty) {
        v = 1;
    }
    if (tx < matrix->columns && ty < matrix->rows) {
        if (NULL != matrix->reValues) {
            matrix->reValues[ty * matrix->columns + tx] = v;
        }
        if (NULL != matrix->imValues) {
            matrix->imValues[ty * matrix->columns + tx] = 0;
        }
    }
}

__hostdevice__ floatt CUDA_getReDiagonal(math::Matrix* matrix, intt index) {
    if (matrix->reValues == NULL) {
        return 0;
    }
    return matrix->reValues[index + matrix->columns * index];
}

__hostdevice__ floatt CUDA_getImDiagonal(math::Matrix* matrix, intt index) {
    if (matrix->imValues == NULL) {
        return 0;
    }
    return matrix->imValues[index + matrix->columns * index];
}

__hostdevice__ floatt CUDA_sum(floatt* buffer, uintt count) {
    floatt sum = 0;
    for (uintt fa = 0; fa < count; ++fa) {
        sum += buffer[fa];
    }
    return sum;
}

#endif	/* DEVICE_H */