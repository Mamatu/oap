/* 
 * File:   CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUCOMMONUTILS_H
#define	CUCOMMONUTILS_H

#include "cuda.h"

#define GetMatrixXIndex(threadIdx, blockIdx, blockDim) (blockIdx.x * blockDim.x + threadIdx.x * 2)

#define GetMatrixYIndex(threadIdx, blockIdx, blockDim) (blockIdx.y * blockDim.y + threadIdx.y)

#define GetMatrixIndex(threadIdx, blockIdx, blockDim) ((threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x * 2))

#define GetLength(blockIdx, blockDim, limit) blockDim - ((blockIdx + 1) * blockDim > limit ? (blockIdx + 1) * blockDim - limit : 0);

#define CalculateSumStep(buffer, index, length)\
{\
    if (index < length / 2) {\
        uintt c = length & 1;\
        buffer[index] += buffer[index + length / 2];\
        if (c == 1 && index + length / 2 == length - 2) {\
            buffer[index] += buffer[length - 1];\
        }\
    }\
}


#define CompareMatrix(matrix, xlength, code, code1)\
const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < matrix->rows\
    && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < matrix->columns;\
if (inScope) {\
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim);\
    uintt c = xlength & 1;\
    code;\
    if (c == 1 && threadIdx.x == xlength / 2 - 1) {\
        code1;\
    }\
}

#endif	/* CUCOMMONUTILS_H */
