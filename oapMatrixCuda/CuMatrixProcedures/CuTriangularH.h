/*
 * Copyright 2016 Marcin Matula
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
#include "CuMatrixProcedures/CuDotProductOptProcedures.h"
#include "CuMatrixProcedures/CuQRProcedures.h"
#include "CuMatrixProcedures/CuIsUpperTriangularProcedures.h"

__hostdevice__ void CUDA_HMtoUTM(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R, math::Matrix* Qoutput,
    math::Matrix* temp1, math::Matrix* temp2, math::Matrix* temp3,
    math::Matrix* temp4, math::Matrix* temp5) {
  HOST_INIT();

  bool status = false;
  CUDA_SetIdentityMatrix(Qoutput);
  status = CUDA_isUpperTriangular(H);
  uintt fa = 0;
  while (status == false) {
    CUDA_QRGR(Q, R, H, temp2, temp3, temp4, temp5);
    CUDA_dotProduct(H, R, Q);
    CUDA_dotProduct(temp1, Q, Qoutput);
    CUDA_switchPointer(&temp1, &Qoutput);
    status = CUDA_isUpperTriangular(H);
    // threads_sync();
    ++fa;
  }
  // TODO: optymalization
  if (fa % 2 == 0) {
    CUDA_copyMatrix(Q, Qoutput);
  } else {
    CUDA_copyMatrix(Q, temp1);
  }
  // cuda_debug_function();
}

#endif  // CUTRIANGULARH
