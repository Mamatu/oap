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



#ifndef CUINVERSIONPROCEDURES
#define CUINVERSIONPROCEDURES

#include "CuIdentityProcedures.h"
#include "CuAdditionProcedures.h"
#include "CuSubstractionProcedures.h"
#include "CuDotProductSpecificProcedures.h"
#include "CuSwitchPointer.h"

__hostdevice__ void CUDA_invertMatrix (math::Matrix* AI, math::Matrix* A, math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3)
{
  HOST_INIT();

  CUDA_SetIdentityMatrix(aux2);
  CUDA_SetIdentityMatrix(aux3);
  CUDA_substractMatrices(aux3, aux2, A);

  for (uintt fa = 0; fa < 15; ++fa)
  {
    CUDA_SetIdentityMatrix(aux2);
    for (uintt fb = 0; fb < fa; ++fb)
    {
      CUDA_specific_dotProduct(aux1, aux2, aux3);
      CUDA_switchPointer(&aux1, &aux2);
    }
    if (fa % 2 != 0)
    {
      CUDA_switchPointer(&aux1, &aux2);
    }
    CUDA_addMatrix(AI, AI, aux1);
  }
}

#endif  // CUINVERSEMATRIX
