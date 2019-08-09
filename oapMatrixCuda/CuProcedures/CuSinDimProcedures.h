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

#ifndef CU_SIN_DIM_PROCEDURES_H
#define CU_SIN_DIM_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuSinProcedures.h"

__hostdeviceinline__ void CUDA_sinDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    if (isre && isim)
    {
      CUDA_sinReal (omatrix, imatrix);
    }
    else if (isre)
    {
      CUDA_sinRe (omatrix, imatrix);
    }
    else if (isim)
    {
      CUDA_sinIm (omatrix, imatrix);
    }
  }
}

__hostdeviceinline__ void CUDA_sinDimDerivative (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    if (isre && isim)
    {
      CUDA_sinDerivativeReal (omatrix, imatrix);
    }
    else if (isre)
    {
      CUDA_sinDerivativeRe (omatrix, imatrix);
    } 
    else if (isim)
    {
      CUDA_sinDerivativeIm (omatrix, imatrix);
    }
  }
}

__hostdeviceinline__ void CUDA_multiplySinDimDerivative (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    if (isre && isim) {
      CUDA_multiplySinDerivativeReal (omatrix, imatrix);
    }
    else if (isre)
    {
      CUDA_multiplySinDerivativeRe (omatrix, imatrix);
    }
    else if (isim)
    {
      CUDA_multiplySinDerivativeIm (omatrix, imatrix);
    }
  }
}

#endif /* CU_SIN_DIM_PROCEDURES_H */
