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



#ifndef CUDOTPRODUCTOPTUTILS
#define CUDOTPRODUCTOPTUTILS

#include "CuCore.h"
#include "Matrix.h"

__hostdeviceinline__ void setSharedMatrixReal(
    floatt* buffer1Re, floatt* buffer1Im, floatt* buffer2Re, floatt* buffer2Im,
    math::Matrix* params0, math::Matrix* params1, uintt offset,
    uintt threadIndexX, uintt threadIndexY) {
  HOST_INIT();
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    buffer1Re[fa1 + blockDim.x * threadIdx.y] =
        params0->reValues[fa1 + columns1 * threadIndexY];
    buffer2Re[fa1 * columns2 + threadIdx.x] =
        params1->reValues[fa1 * columns2 + threadIndexX];
    buffer1Im[fa1 + blockDim.x * threadIdx.y] =
        params0->imValues[fa1 + columns1 * threadIndexY];
    buffer2Im[fa1 * columns2 + threadIdx.x] =
        params1->imValues[fa1 * columns2 + threadIndexX];
  }
}

__hostdeviceinline__ void setSharedMatrixRe(floatt* buffer1Re,
                                            floatt* buffer2Re,
                                            math::Matrix* params0,
                                            math::Matrix* params1, uintt offset,
                                            uintt threadIndexX,
                                            uintt threadIndexY) {}

__hostdeviceinline__ void setSharedMatrixIm(floatt* buffer1Im,
                                            floatt* buffer2Im,
                                            math::Matrix* params0,
                                            math::Matrix* params1, uintt offset,
                                            uintt threadIndexX,
                                            uintt threadIndexY) {}

#endif  // CUDOTPRODUCTOPTUTILS
