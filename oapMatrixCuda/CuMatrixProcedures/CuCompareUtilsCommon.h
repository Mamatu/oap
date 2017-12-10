/*
 * Copyright 2016, 2017 Marcin Matula
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

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include <math.h>

#define COMPARE_LIMIT 0.0001f

//__hostdevice__ floatt cuda_getRealDist(math::Matrix* m1, math::Matrix* m2, uint column, uint row) {
//  return cuda_getRealDist(m1, m2, column + m1->columns * row);
//}

__hostdevice__ floatt cuda_getRealDist(math::Matrix* m1, math::Matrix* m2, uint index) {
  floatt re1 = m1->reValues[index];
  floatt im1 = m1->imValues[index];
  floatt re2 = m2->reValues[index];
  floatt im2 = m2->imValues[index];
  return sqrtf((re1-re2)*(re1-re2) - (im1-im2)*(im1-im2) - 2*re1*re2*im1*im2);
}

__hostdevice__ floatt cuda_getReDist(math::Matrix* m1, math::Matrix* m2, uint index) {
  return sqrtf((m1->reValues[index] - m2->reValues[index]) * (m1->reValues[index] - m2->reValues[index]));
}

__hostdevice__ floatt cuda_getImDist(math::Matrix* m1, math::Matrix* m2, uint index) {
  return sqrtf(-1 * (m1->imValues[index] - m2->imValues[index]) * (m1->imValues[index] - m2->imValues[index]));
}

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
