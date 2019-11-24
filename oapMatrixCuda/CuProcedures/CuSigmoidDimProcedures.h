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

#ifndef CU_SIGMOID_DIM_PROCEDURES_H
#define CU_SIGMOID_DIM_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuSigmoidProcedures.h"

__hostdeviceinline__ void cuda_sigmoidDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_sigmoid (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_dsigmoidDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_dsigmoid (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_multiplyDSigmoidDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_multiplyDSigmoid (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_sigmoidDimPeriodic (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_sigmoid (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_dsigmoidDimPeriodic (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_dsigmoid (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_multiplyDSigmoidDimPeriodic (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_multiplyDSigmoid (omatrix, imatrix);
  }
}

#endif
