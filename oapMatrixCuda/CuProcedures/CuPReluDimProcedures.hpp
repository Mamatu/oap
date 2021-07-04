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

#ifndef CU_PRELU_DIM_PROCEDURES_H
#define CU_PRELU_DIM_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "CuPReluProcedures.hpp"

__hostdeviceinline__ void cuda_preluDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];
  floatt alpha = 0.01;

  if (isInRange)
  {
    cuda_prelu_alpha (omatrix, imatrix, alpha);
  }
}

__hostdeviceinline__ void cuda_dpreluDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];
  floatt alpha = 0.01;

  if (isInRange)
  {
    cuda_dprelu_alpha (omatrix, imatrix, alpha);
  }
}

__hostdeviceinline__ void cuda_preluDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);
  floatt alpha = 0.01;

  if (isInRange)
  {
    cuda_prelu_alpha (omatrix, imatrix, alpha);
  }
}

__hostdeviceinline__ void cuda_dpreluDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex)
{
  bool isInRange = cuda_inRangePD(omatrix, ex);
  floatt alpha = 0.01;

  if (isInRange)
  {
    cuda_dprelu_alpha (omatrix, imatrix, alpha);
  }
}

#endif /* CU_RELU_DIM_PROCEDURES_H */
