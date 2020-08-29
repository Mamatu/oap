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

#ifndef CU_TANH_DIM_PROCEDURES_H
#define CU_TANH_DIM_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuTanhProcedures.h"

__hostdeviceinline__ void cuda_tanhDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_tanh (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_dtanhDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_dtanh (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_multiplyDTanhDim (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_multiplyDTanh (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_tanhDimPeriodic (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_tanh (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_dtanhDimPeriodic (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_dtanh (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_multiplyDTanhDimPeriodic (math::Matrix* omatrix, math::Matrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_multiplyDTanh (omatrix, imatrix);
  }
}

#endif
