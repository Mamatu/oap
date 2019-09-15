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

#ifndef CU_VECTOR_UTILS_H
#define CU_VECTOR_UTILS_H

#include "CuCore.h"
#include "Matrix.h"
#include "Logger.h"

__hostdevice__ void CUDAKernel_setVector(math::Matrix* V, uintt column,
                                   math::Matrix* v, uintt length) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length) {
    uintt index1 = threadIndexY * V->columns + column + threadIndexX;
    uintt index2 = threadIndexY * v->columns + threadIndexX;
    if (V->reValues != NULL && v->reValues != NULL) {
      V->reValues[index1] = v->reValues[index2];
    }
    if (V->imValues != NULL && v->imValues != NULL) {
      V->imValues[index1] = v->imValues[index2];
    }
  }
  threads_sync();
}

__hostdevice__ void CUDAKernel_getVector(math::Matrix* v, uintt length,
                                   math::Matrix* V, uintt column) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length) {
    uintt index1 = threadIndexY * V->columns + column + threadIndexX;
    uintt index2 = threadIndexY * v->columns + threadIndexX;
    if (V->reValues != NULL && v->reValues != NULL) {
      v->reValues[index2] = V->reValues[index1];
    }
    if (V->imValues != NULL && v->imValues != NULL) {
      v->imValues[index2] = V->imValues[index1];
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_setVector(math::Matrix* V, uintt column,
                                   math::Matrix* v, uintt length)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length && threadIndexX >= column && threadIndexX < column + v->columns)
  {
    uintt index1 = threadIndexY * V->columns + threadIndexX;
    uintt index2 = threadIndexY * v->columns + (threadIndexX - column);
    if (V->reValues != NULL && v->reValues != NULL)
    {
      V->reValues[index1] = v->reValues[index2];
    }
    if (V->imValues != NULL && v->imValues != NULL)
    {
      V->imValues[index1] = v->imValues[index2];
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_getVector(math::Matrix* v, uintt length,
                                   math::Matrix* V, uintt column) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length && threadIndexX >= column && threadIndexX < column + v->columns)
  {
    uintt index1 = threadIndexY * V->columns + threadIndexX;
    uintt index2 = threadIndexY * v->columns + (threadIndexX - column);

#ifndef OAP_CUDA_BUILD
    debugAssert (index1 < V->columns * V->rows);
    debugAssert (index2 < v->columns * v->rows);
#endif

    if (V->reValues != NULL && v->reValues != NULL)
    {
      v->reValues[index2] = V->reValues[index1];
    }
    if (V->imValues != NULL && v->imValues != NULL)
    {
      v->imValues[index2] = V->imValues[index1];
    }
  }
  threads_sync();
}

#endif /* CU_VECTOR_UTILS_H */
