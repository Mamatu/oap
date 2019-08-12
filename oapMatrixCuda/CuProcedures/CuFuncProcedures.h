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

#ifndef CU_FUNC_PROCEDURES_H
#define CU_FUNC_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

typedef void(*func_t)(floatt*, floatt);

__hostdeviceinline__ void CUDA_funcRe (math::Matrix* omatrix, math::Matrix* imatrix, func_t func)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* output = &omatrix->reValues[index];
  func (output, imatrix->reValues[index]);
}

__hostdeviceinline__ void CUDA_funcIm (math::Matrix* omatrix, math::Matrix* imatrix, func_t func)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  func (output, imatrix->imValues[index]);
}

__hostdeviceinline__ void CUDA_funcReal (math::Matrix* omatrix, math::Matrix* imatrix, func_t func)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  func (reoutput, imatrix->reValues[index]);
  func (imoutput, imatrix->imValues[index]);
}

__hostdeviceinline__ void CUDA_func (math::Matrix* omatrix, math::Matrix* imatrix, func_t func)
{
  HOST_INIT();

  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;

  if (isre && isim)
  {
    CUDA_funcReal (omatrix, imatrix, func);
  }
  else if (isre)
  {
    CUDA_funcRe (omatrix, imatrix, func);
  }
  else if (isim)
  {
    CUDA_funcIm (omatrix, imatrix, func);
  }
}

__hostdeviceinline__ bool cuda_inRangePD (math::Matrix* matrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt indexY1 = (threadIndexY) % ex[2];
  return threadIndexX < ex[0] && threadIndexY < matrix->rows && indexY1 < ex[1];
}

#endif
