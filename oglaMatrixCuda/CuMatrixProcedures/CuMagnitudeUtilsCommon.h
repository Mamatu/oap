/* 
 * File: CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUMAGNITUDEUTILSCOMMON_H
#define	CUMAGNITUDEUTILSCOMMON_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"

#define GetMatrixXIndex2(threadIdx, blockIdx, blockDim) ((blockIdx.x * blockDim.x + threadIdx.x) * 2)

#define GetMatrixXIndex(threadIdx, blockIdx, blockDim) ((blockIdx.x * blockDim.x + threadIdx.x))

#define GetMatrixYIndex(threadIdx, blockIdx, blockDim) (blockIdx.y * blockDim.y + threadIdx.y)

#define GetMatrixIndex(threadIdx, blockIdx, blockDim, offset) (GetMatrixYIndex(threadIdx, blockIdx, blockDim) * (offset) + (GetMatrixXIndex(threadIdx, blockIdx, blockDim)))

#define GetMatrixIndex2(threadIdx, blockIdx, blockDim, offset) (GetMatrixYIndex(threadIdx, blockIdx, blockDim) * (offset) + (GetMatrixXIndex2(threadIdx, blockIdx, blockDim)))

#define GetLength(blockIdx, blockDim, limit) blockDim - ((blockIdx + 1) * blockDim > limit ? (blockIdx + 1) * blockDim - limit : 0);

#endif	/* CUCOMMONUTILS_H */
