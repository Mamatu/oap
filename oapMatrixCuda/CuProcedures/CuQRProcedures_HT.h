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

#ifndef CUQRPROCEDURES_HT_H
#define CUQRPROCEDURES_HT_H

#include "CuCore.h"
#include "CuUtils.h"
#include "MatrixAPI.h"
#include "CuCopyProcedures.h"
#include "CuAdditionProcedures.h"
#include "CuTransposeProcedures.h"
#include "CuDotProductProcedures.h"
#include "CuSubstractionProcedures.h"
#include "CuIdentityProcedures.h"
#include "CuMagnitudeUtils.h"
#include "CuMagnitudeOptProcedures.h"
#include "CuMultiplicationProcedures.h"
#include "CuVectorUtils.h"
#include "CuIdentityMatrixOperations.h"
#include "CuSubstractionProcedures.h"
#include "CuSetMatrixProcedures.h"
#include "CuSwitchPointer.h"


#ifndef OAP_CUDA_BUILD
#include "oapHostMatrixUtils.h"
#endif
//#define cuda_debug(x, ...)
//#define cuda_debug_buffer(buff, len)


__hostdeviceinline__ floatt sign (floatt x)
{
  if (x == 0)
  {
    return 1.;
  }
  return x < 0 ? -1. : 1.;
}

__hostdeviceinline__ floatt csign (floatt x)
{
  return sign (x);
}

__hostdevice__ void CUDA_QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, floatt* buffer, math::Matrix* P, math::Matrix* VVT)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt n = A->rows; 

  for (uint k = 0; k < n; ++k)
  {
    uintt columnIdx = k;
    uintt rowIdx = k;

    math::Matrix* M = A;

    if (k > 0)
    {
      M = R;
    }

    floatt aXY = GetRe (M, columnIdx, rowIdx);

    int j = threadIndexY;
    floatt sum = CUDA_calcMagnitudeOptEx (M, buffer, columnIdx, k, 1, n - k);

    threads_sync ();

    CUDA_getVector (V, V->rows, M, columnIdx);

    if (threadIndexX == 0 && threadIndexY == 0)
    {
      for (uintt rIdx = 0; rIdx < rowIdx; ++rIdx)
      {
        SetRe (V, 0, rIdx, 0.f);
      }
      SetRe (V, 0, rowIdx, GetRe(V, 0, rowIdx) + csign(GetRe(M, columnIdx, rowIdx)) * sqrtf(sum));
    }
    threads_sync();

    CUDA_transposeMatrix (VT, V);
    CUDA_dotProduct (VVT, V, VT);

    sum = CUDA_calcMagnitudeOptEx (V, buffer, 0, 0, 1, V->rows);
    if (sum != 0)
    {
      CUDA_multiplyConstantMatrix (VVT, VVT, 2. / sum, 0.);
      CUDA_IdentityMatrixSubstract (P, VVT);

      CUDA_copyMatrix (VVT, M);

      CUDA_dotProduct (R, P, VVT);
    }
    if (k == 0)
    {
      CUDA_copyMatrix (Q, P);
    }
    else
    {
      CUDA_copyMatrix (VVT, Q);
      CUDA_dotProduct (Q, VVT, P);
    }
  }
}

__hostdevice__ void CudaKernel_QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT)
{
  floatt* sharedMemory = NULL;
  HOST_INIT_SHARED (floatt, sharedMemory);
  CUDA_QRHT (Q, R, A, V, VT, sharedMemory, P, VVT);
}

#endif /* CUQRPROCEDURES_H */
