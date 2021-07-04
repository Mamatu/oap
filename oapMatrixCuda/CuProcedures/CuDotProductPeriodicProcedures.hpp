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

#ifndef OAP_CU_DOT_PRODUCT_PERIODIC_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_PERIODIC_PROCEDURES_H

#include "CuDotProductGenericProcedures.hpp"


/**
 * If C = A * B and rows of A are lower than rows of C and
 * columns of A are lower than rows of B fo example:
 *
 *              B00 B01 B02
 *              B10 B11 B12
 *              B20 B21 B22
 *              B30 B31 B32
 *              B40 B41 B42
 *              B50 B51 B52
 *
 * A00 A01 A02  C00 C01 C02
 * A10 A11 A12  C10 C11 C12
 * A20 A21 A22  C20 C21 C22
 *              C30 C31 C32
 *              C40 C41 C42
 *              C50 C51 C52
 *
 * then behaviour of this procedure is
 *
 *              B00 B01 B02
 *              B10 B11 B12
 *              B20 B21 B22
 *              B30 B31 B32
 *              B40 B41 B42
 *              B50 B51 B52
 *
 * A00 A01 A02  C00 C01 C02
 * A10 A11 A12  C10 C11 C12
 * A20 A21 A22  C20 C21 C22
 * A00 A01 A02  C30 C31 C32
 * A10 A11 A12  C40 C41 C42
 * A20 A21 A22  C50 C51 C52
 *
 * so for example
 *
 * C10 = A10 * B00 + A11 * B10 + A12 * B30
 * C40 = A10 * B30 + A11 * B40 + A12 * B50
 *
 */
__hostdevice__ void CUDA_dotProductPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;

  bool inRange = threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  
  uintt offset = gColumns (params0);
  uintt indexY1 = (threadIndexY) % offset;

  uintt t0[2] = {0, indexY1};
  uintt t1[2] = {threadIndexX, (threadIndexY / offset) * offset};

  cuda_generic_dotProductUserThreads (output, params0, params1, t0, t1, offset, inRange);
}

#endif
