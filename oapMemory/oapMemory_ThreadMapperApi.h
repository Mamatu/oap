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
#include "MatrixAPI.h"
#include "oapThreadsMapperS.h"
#include "oapMemory_CommonApi.h"
#include "oapMemory_ThreadMapperApi_AbsIndexAlgo.h"
#include "oapMemory_ThreadMapperApi_MatrixPosAlgo.h"

namespace oap
{
namespace threads
{

__hostdeviceinline__ void GetIdx (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, uintt out[2], const math::Matrix* const* arg, const ThreadsMapperS* mapper, uintt argIdx)
{
  switch (mapper->mode)
  {
    case OAP_THREADS_MAPPER_MODE__AIA:
        aia::GetIdx_AbsIndexAlgo (threadIdx, blockIdx, blockDim, gridDim, out, arg, mapper, argIdx);
      break;

    case OAP_THREADS_MAPPER_MODE__MP:
        uintt out1[3];
        mp::GetIdx_MatrixPosAlgo (threadIdx, blockIdx, blockDim, gridDim, out1, arg, mapper, argIdx);
        out[0] = out1[0];
        out[1] = out1[1] + GetColumns(arg[out[0]]) * out1[2];
      break;

    default:
        assert ("Not supported" != NULL);
      break;
  };
}

__hostdeviceinline__ void GetPos (dim3 threadIdx, dim3 blockIdx, dim3 blockDim, dim3 gridDim, uintt out[2], const math::Matrix* const* arg, const ThreadsMapperS* mapper, uintt argIdx)
{
  switch (mapper->mode)
  {
    case OAP_THREADS_MAPPER_MODE__AIA:
        assert ("Not supported" != NULL);
      break;

    case OAP_THREADS_MAPPER_MODE__MP:
        mp::GetIdx_MatrixPosAlgo (threadIdx, blockIdx, blockDim, gridDim, out, arg, mapper, argIdx);
      break;

    default:
        assert ("Not supported" != NULL);
      break;
  };
}

}
}

#define _idxs(out, matrices, mapper, argIdx) oap::threads::GetIdx(threadIdx, blockIdx, blockDim, gridDim, out, matrices, mapper, argIdx)

#define _idxpos(out, matrices, mapper, argIdx) oap::threads::GetPos(threadIdx, blockIdx, blockDim, gridDim, out, matrices, mapper, argIdx)

#endif
