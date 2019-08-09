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

#ifndef CU_TANH_DIM_PROCEDURES_H
#define CU_TANH_DIM_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuTanhProcedures.h"

__hostdeviceinline__ void CUDA_tanhDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
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
      CUDA_tanhReal (omatrix, imatrix);
    }
    else if (isre)
    {
      CUDA_tanhRe (omatrix, imatrix);
    }
    else if (isim)
    {
      CUDA_tanhIm (omatrix, imatrix);
    }
  }
}

__hostdeviceinline__ void CUDA_tanhDimDerivative(math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
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
      CUDA_tanhDerivativeReal (omatrix, imatrix);
    }
    else if (isre)
    {
      CUDA_tanhDerivativeRe (omatrix, imatrix);
    }
    else if (isim)
    {
      CUDA_tanhDerivativeIm (omatrix, imatrix);
    }
  }
}

__hostdeviceinline__ void CUDA_multiplyTanhDimDerivative(math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
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
      CUDA_multiplyTanhDerivativeReal (omatrix, imatrix);
    }
    else if (isre)
    {
      CUDA_multiplyTanhDerivativeRe (omatrix, imatrix);
    }
    else if (isim)
    {
      CUDA_multiplyTanhDerivativeIm (omatrix, imatrix);
    }
  }
}

#endif /* CU_TANH_DIM_PROCEDURES_H */
