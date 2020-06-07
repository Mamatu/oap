/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef OAP_MEMORY__THREAD_MAPPER_API_H
#define OAP_MEMORY__THREAD_MAPPER_API_H

#include "Matrix.h"
#include "oapMemory_CommonApi.h"
#include "oapThreadMapperPrimitives.h"
#if 0
namespace oap
{
namespace threads
{

__hostdeviceinline__ void GetMapping_RelativeSharedPointers (dim3& memoryIdx, dim3& matrixIdx, const dim3& threadIdx, const dim3& blockIdx, const dim3& blockDim, const dim3& gridDim, const ThreadsMapper* mapper)
{
/*  math::Matrix** matrices = static_cast<math::Matrix**>(mapper->data);
  math::Matrix* matrixPtr = matrices[threadIdx.y * blockDim.x + threadIdx.x];
  matrixIdx.x = matrixPtr->loc.x;
  matrixIdx.y = matrixPtr->loc.y;
  memoryIdx.x = matrixIdx.x + threadIdx.x;
  memoryIdx.y = matrixIdx.y + threadIdx.y;*/
}

__hostdeviceinline__ void GetMapping (dim3& memoryIdx, dim3& matrixIdx, const dim3& threadIdx, const dim3& blockIdx, const dim3& blockDim, const dim3& gridDim, const ThreadsMapper* mapper)
{
  if (mapper->mode == OAP_MAPPER_MODE__RELATIVE_SHARED_POINTERS)
  {
    GetMapping_RelativeSharedPointers (memoryIdx, matrixIdx, threadIdx, blockIdx, blockDim, gridDim, mapper);
  }
}

}
}

#define oap_CALCULATE_MEM_INDEX(dim) blockIdx.dim * blockDim.dim + threadIdx.dim;

#define OAP_THREADS_MAPPER(mapper)                                                      \
  dim3 memoryIdx;                                                                       \
  dim3 matrixIdx;                                                                       \
  if (mapper == NULL)                                                                   \
  {                                                                                     \
    memoryIdx.x = oap_CALCULATE_MEM_INDEX(x);                                           \
    memoryIdx.y = oap_CALCULATE_MEM_INDEX(y);                                           \
    matrixIdx = memoryIdx;                                                              \
  }                                                                                     \
  else                                                                                  \
  {                                                                                     \
    GetMapping (memoryIdx, matrixIdx, threadIdx, blockIdx, blockDim, gridDim, mapper);  \
  }

#endif
#endif
