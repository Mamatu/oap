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



#ifndef CUCOMPAREUTILSCOMMON
#define CUCOMPAREUTILSCOMMON

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include <math.h>

#define COMPARE_LIMIT 0.0001f

__hostdevice__ int cuda_isEqualReIndex(math::Matrix* matrix1,
                                       math::Matrix* matrix2, uintt index) {
  HOST_INIT();
  return fabs(matrix1->reValues[index] - matrix2->reValues[index]) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualImIndex(math::Matrix* matrix1,
                                       math::Matrix* matrix2, uintt index) {
  HOST_INIT();
  return fabs(matrix1->imValues[index] - matrix2->imValues[index]) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualRe(math::Matrix* matrix1, math::Matrix* matrix2,
                                  uintt column, uintt row) {
  HOST_INIT();
  return fabs(GetRe(matrix1, column, row) - GetRe(matrix2, column, row)) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualIm(math::Matrix* matrix1, math::Matrix* matrix2,
                                  uintt column, uintt row) {
  HOST_INIT();
  return fabs(GetIm(matrix1, column, row) - GetIm(matrix2, column, row)) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

#endif  // CUCOMPAREUTILSCOMMON
