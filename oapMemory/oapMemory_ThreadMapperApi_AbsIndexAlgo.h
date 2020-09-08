/*
 * Copyright 2016 - 2021 Marcin Matula
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
#include "oapMemory_ThreadMapperApi_Types.h"

namespace oap
{
namespace aia
{

__hostdevice__ void GetIdx_AbsIndexAlgo (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, uintt out[2], const math::Matrix* const* arg, const oap::ThreadsMapperS* mapper, uintt argIdx)
{
  oap::threads::UserData* ud = static_cast<oap::threads::UserData*>(mapper->data);

  uintt idx = ud->mapperBuffer [_cuGlbThreadIdx()];
  idx += argIdx * AIA_INDECIES_COUNT;

  out[0] = ud->dataBuffer[idx];
  out[1] = ud->dataBuffer[idx + 1];
}

__hostdevice__ bool GetIdxCheck_AbsIndexAlgo (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, uintt out[2], const math::Matrix* const* arg, const oap::ThreadsMapperS* mapper, uintt argIdx)
{
  oap::threads::UserData* ud = static_cast<oap::threads::UserData*>(mapper->data);

  uintt idx = ud->mapperBuffer [_cuGlbThreadIdx()];

  if (idx != MAX_UINTT)
  {
    idx += argIdx * AIA_INDECIES_COUNT;

    out[0] = ud->dataBuffer[idx];
    out[1] = ud->dataBuffer[idx + 1];
    return true;
  }
  return false;
}

__hostdeviceinline__ bool InRange_AbsIndexAlgo (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, const oap::ThreadsMapperS* mapper)
{
  oap::threads::UserData* ud = static_cast<oap::threads::UserData*>(mapper->data);
  uintt idx = ud->mapperBuffer [_cuGlbThreadIdx()];
  return idx != MAX_UINTT;
}

}
}

#endif
