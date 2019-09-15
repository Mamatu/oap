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



#ifndef CUTRIANGULARH
#define CUTRIANGULARH

#include "CuCore.h"
#include "Matrix.h"
#include "CuProcedures/CuDotProductOptProcedures.h"
#include "CuProcedures/CuQRProcedures_GR.h"
#include "CuProcedures/CuIsUpperTriangularProcedures.h"

#define CUDA_HMtoUTMStep_STEPS 1000

__hostdevice__ void CUDA_HMtoUTM(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R,
    math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
    math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  HOST_INIT();

  bool status = false;
  CUDA_SetIdentityMatrix(aux1);
  status = CUDA_isUpperTriangular(H);
  uint fa = 0;
  for (; fa < CUDA_HMtoUTMStep_STEPS && status == false; ++fa) {
    CUDA_QRGR(Q, R, H, aux3, aux4, aux5, aux6);
    CUDA_dotProduct(H, R, Q);
    CUDA_dotProduct(aux2, Q, aux1);
    CUDA_switchPointer(&aux2, &aux1);
    status = CUDA_isUpperTriangular(H);
  }
  // TODO: optymalization
  if (fa % 2 == 0) {
    CUDA_copyMatrix(Q, aux1);
  } else {
    CUDA_copyMatrix(Q, aux2);
  }
}

__hostdevice__ void CUDA_HMtoUTMStep(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R,
    math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
    math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  HOST_INIT();

  bool status = false;
  CUDA_SetIdentityMatrix(aux1);
  uint fa = 0;
  status = CUDA_isUpperTriangular(H);
  for (; fa < CUDA_HMtoUTMStep_STEPS && status == false; ++fa) {
    CUDA_QRGR(Q, R, H, aux3, aux4, aux5, aux6);
    CUDA_dotProduct(H, R, Q);
    CUDA_dotProduct(aux2, Q, aux1);
    CUDA_switchPointer(&aux2, &aux1);
    status = CUDA_isUpperTriangular(H);
  }
  // TODO: optymalization
  if (fa % 2 == 0) {
    CUDA_copyMatrix(Q, aux1);
  } else {
    CUDA_copyMatrix(Q, aux2);
  }
}

#endif  // CUTRIANGULARH
