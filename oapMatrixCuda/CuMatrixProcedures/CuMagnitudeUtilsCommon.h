/*
 * Copyright 2016, 2017 Marcin Matula
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




#ifndef CUMAGNITUDEUTILSCOMMON_H
#define CUMAGNITUDEUTILSCOMMON_H

#include "CuCore.h"
#include "Matrix.h"

#define GetMatrixXIndex2(threadIdx, blockIdx, blockDim) \
  ((blockIdx.x * blockDim.x + threadIdx.x) * 2)

#define GetMatrixXIndex(threadIdx, blockIdx, blockDim) \
  ((blockIdx.x * blockDim.x + threadIdx.x))

#define GetMatrixYIndex(threadIdx, blockIdx, blockDim) \
  (blockIdx.y * blockDim.y + threadIdx.y)

#define GetMatrixIndex(threadIdx, blockIdx, blockDim, offset)  \
  (GetMatrixYIndex(threadIdx, blockIdx, blockDim) * (offset) + \
   (GetMatrixXIndex(threadIdx, blockIdx, blockDim)))

#define GetMatrixIndex2(threadIdx, blockIdx, blockDim, offset) \
  (GetMatrixYIndex(threadIdx, blockIdx, blockDim) * (offset) + \
   (GetMatrixXIndex2(threadIdx, blockIdx, blockDim)))

#define GetLength(blockIdx, blockDim, limit)          \
  blockDim - ((blockIdx + 1) * blockDim > limit       \
                  ? (blockIdx + 1) * blockDim - limit \
                  : 0);

#endif /* CUCOMMONUTILS_H */
