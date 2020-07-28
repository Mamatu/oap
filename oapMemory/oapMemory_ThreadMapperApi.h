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
#include "oapThreadsMapperS.h"
#include "oapMemory_CommonApi.h"
#include "oapMemory_ThreadMapperApi_AbsIndexAlgo.h"

namespace oap
{
namespace threads
{

__hostdeviceinline__ uintt GetIdx (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, const Memory& memory, const MemoryRegion& region, const ThreadsMapperS* mapper, uintt argIdx)
{
  switch (mapper->mode)
  {
    case OAP_THREADS_MAPPER_MODE__SIMPLE:
      return aia::GetIdx_AbsIndexAlgo (threadIdx, blockIdx, blockDim, gridDim, memory, region, mapper, argIdx);

    default:
      return aia::GetIdx_AbsIndexAlgo (threadIdx, blockIdx, blockDim, gridDim, memory, region, mapper, argIdx);
  };
}

__hostdeviceinline__ bool InRange (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, const ThreadsMapperS* mapper)
{
  switch (mapper->mode)
  {
    case OAP_THREADS_MAPPER_MODE__SIMPLE:
      return aia::InRange_AbsIndexAlgo (threadIdx, blockIdx, blockDim, gridDim, mapper);

    default:
      return aia::InRange_AbsIndexAlgo (threadIdx, blockIdx, blockDim, gridDim, mapper);
  };
}

}
}

#define _idx(memory, region, mapper, argIdx) oap::threads::GetIdx(threadIdx, blockIdx, blockDim, gridDim, memory, region, mapper, argIdx)

#define _inRange(mapper) oap::threads::InRange(threadIdx, blockIdx, blockDim, gridDim, mapper) 

#endif
