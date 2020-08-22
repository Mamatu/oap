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

#ifndef OAP_MEMORY__THREAD_MAPPER_API__ABS_INDEX_ALGO_H
#define OAP_MEMORY__THREAD_MAPPER_API__ABS_INDEX_ALGO_H

#include "Matrix.h"
#include "oapThreadsMapperS.h"
#include "oapMemory_CommonApi.h"
#include "oapMemory_ThreadMapperApi_AbsIndexAlgo_CommonApi.h"

namespace oap
{
namespace aia
{

namespace
{
__hostdeviceinline__ uintt getDataIdx (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, const oap::ThreadsMapperS* mapper)
{
  const uintt x = _memoryIdxX();
  const uintt y = _memoryIdxY();
  UserData* ud = static_cast<UserData*>(mapper->data);

  return (y * _memoryWidth() + x) * (ud->argsCount * INDECIES_COUNT);
}
}

__hostdevice__ void GetIdx_AbsIndexAlgo (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, uintt out[2], const math::Matrix* const* arg, const oap::ThreadsMapperS* mapper, uintt argIdx)
{
  UserData* ud = static_cast<UserData*>(mapper->data);
  uintt* indecies = static_cast<uintt*>(ud->buffer);

  uintt idx = getDataIdx (threadIdx, blockIdx, blockDim, gridDim, mapper);

  idx += argIdx * INDECIES_COUNT;

  out[0] = indecies[idx];
  out[1] = indecies[idx + 1];
}

__hostdeviceinline__ bool InRange_AbsIndexAlgo (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, const oap::ThreadsMapperS* mapper)
{
  UserData* ud = static_cast<UserData*>(mapper->data);
  uintt* indecies = static_cast<uintt*>(ud->buffer);

  uintt dataIdx = getDataIdx (threadIdx, blockIdx, blockDim, gridDim, mapper);

  return indecies[dataIdx] != MAX_UINTT;
}

}
}

#endif
