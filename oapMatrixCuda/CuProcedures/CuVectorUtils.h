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

#ifndef CU_VECTOR_UTILS_H
#define CU_VECTOR_UTILS_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "Logger.h"

__hostdevice__ void CUDAKernel_setVector (math::Matrix* V, uintt column,
                                          math::Matrix* v, uintt length)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length)
  {
    uintt index1 = threadIndexY * gColumns (V) + column + threadIndexX;
    uintt index2 = threadIndexY * gColumns (v) + threadIndexX;
    if (V->re.ptr != NULL && v->re.ptr != NULL)
    {
      *GetRePtrIndex (V, index1) = GetReIndex (v, index2);
    }
    if (V->im.ptr != NULL && v->im.ptr != NULL)
    {
      *GetImPtrIndex (V, index1) = GetImIndex (v, index2);
    }
  }
  threads_sync();
}

__hostdevice__ void CUDAKernel_getVector (math::Matrix* v, uintt length,
                                          math::Matrix* V, uintt column)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length)
  {
    uintt index1 = threadIndexY * gColumns (V) + column + threadIndexX;
    uintt index2 = threadIndexY * gColumns (v) + threadIndexX;
    if (V->re.ptr != NULL && v->re.ptr != NULL)
    {
      *GetRePtrIndex (v, index2) = GetReIndex (V, index1);
    }
    if (V->im.ptr != NULL && v->im.ptr != NULL)
    {
      *GetImPtrIndex (v, index2) = GetImIndex (V, index1);
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_setVector (math::Matrix* V, uintt column,
                                    math::Matrix* v, uintt length)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length && threadIndexX >= column && threadIndexX < column + gColumns (v))
  {
    uintt index1 = threadIndexY * gColumns (V) + threadIndexX;
    uintt index2 = threadIndexY * gColumns (v) + (threadIndexX - column);
    if (V->re.ptr != NULL && v->re.ptr != NULL)
    {
      *GetRePtrIndex (V, index1) = GetReIndex (v, index2);
    }
    if (V->im.ptr != NULL && v->im.ptr != NULL)
    {
      *GetImPtrIndex (V, index1) = GetImIndex (v, index2);
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_getVector (math::Matrix* v, uintt length,
                                    math::Matrix* V, uintt column)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length && threadIndexX >= column && threadIndexX < column + gColumns (v))
  {
    uintt index1 = threadIndexY * gColumns (V) + threadIndexX;
    uintt index2 = threadIndexY * gColumns (v) + (threadIndexX - column);

#ifndef OAP_CUDA_BUILD
    debugAssert (index1 < gColumns (V) * gRows (V));
    debugAssert (index2 < gColumns (v) * gRows (v));
#endif

    if (V->re.ptr != NULL && v->re.ptr != NULL)
    {
      *GetRePtrIndex (v, index2) = GetReIndex (V, index1);
    }
    if (V->im.ptr != NULL && v->im.ptr != NULL)
    {
      *GetImPtrIndex (v, index2) = GetImIndex (V, index1);
    }
  }
  threads_sync();
}

#endif /* CU_VECTOR_UTILS_H */
