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

#ifndef CU_SIN_DIM_PROCEDURES_H
#define CU_SIN_DIM_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuSinProcedures.h"

__hostdeviceinline__ void cuda_sinDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_sin (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_dsinDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_dsin (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_multiplyDSinDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isInRange)
  {
    cuda_multiplyDSin (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_sinDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_sin (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_dsinDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_dsin (omatrix, imatrix);
  }
}

__hostdeviceinline__ void cuda_multiplyDSinDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);

  if (isInRange)
  {
    cuda_multiplyDSin (omatrix, imatrix);
  }
}

#endif /* CU_SIN_DIM_PROCEDURES_H */
