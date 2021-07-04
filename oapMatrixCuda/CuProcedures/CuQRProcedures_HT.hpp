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

#ifndef CUQRPROCEDURES_HT_H
#define CUQRPROCEDURES_HT_H

#include "CuCore.hpp"
#include "CuUtils.hpp"
#include "MatrixAPI.hpp"
#include "CuCopyProcedures.hpp"
#include "CuAdditionProcedures.hpp"
#include "CuTransposeProcedures.hpp"
#include "CuDotProductSpecificProcedures.hpp"
#include "CuSubtractionProcedures.hpp"
#include "CuIdentityProcedures.hpp"
#include "CuMagnitudeUtils.hpp"
#include "CuMagnitudeOptProcedures.hpp"
#include "CuMultiplicationProcedures.hpp"
#include "CuVectorUtils.hpp"
#include "CuIdentityMatrixOperations.hpp"
#include "CuSubtractionProcedures.hpp"
#include "CuSetMatrixProcedures.hpp"
#include "CuSwitchPointer.hpp"


#ifndef OAP_CUDA_BUILD
#include "oapHostComplexMatrixApi.hpp"
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

__hostdevice__ void CUDA_QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, floatt* buffer, math::ComplexMatrix* P, math::ComplexMatrix* VVT)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt n = gRows (A); 

  for (uint k = 0; k < n; ++k)
  {
    uintt columnIdx = k;
    uintt rowIdx = k;

    math::ComplexMatrix* M = A;

    if (k > 0)
    {
      M = R;
    }

    floatt aXY = GetRe (M, columnIdx, rowIdx);

    int j = threadIndexY;
    floatt sum = CUDA_calcMagnitudeOptEx (M, buffer, columnIdx, k, 1, n - k);

    threads_sync ();

    CUDA_getVector (V, gRows (V), M, columnIdx);

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
    CUDA_specific_dotProduct (VVT, V, VT);

    sum = CUDA_calcMagnitudeOptEx (V, buffer, 0, 0, 1, gRows (V));
    if (sum != 0)
    {
      CUDA_multiplyConstantMatrix (VVT, VVT, 2. / sum, 0.);
      CUDA_IdentityMatrixSubstract (P, VVT);

      CUDA_copyMatrix (VVT, M);

      CUDA_specific_dotProduct (R, P, VVT);
    }
    if (k == 0)
    {
      CUDA_copyMatrix (Q, P);
    }
    else
    {
      CUDA_copyMatrix (VVT, Q);
      CUDA_specific_dotProduct (Q, VVT, P);
    }
  }
}

__hostdevice__ void CudaKernel_QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, math::ComplexMatrix* P, math::ComplexMatrix* VVT)
{
  floatt* sharedMemory = NULL;
  HOST_INIT_SHARED (floatt, sharedMemory);
  CUDA_QRHT (Q, R, A, V, VT, sharedMemory, P, VVT);
}

#endif /* CUQRPROCEDURES_H */
