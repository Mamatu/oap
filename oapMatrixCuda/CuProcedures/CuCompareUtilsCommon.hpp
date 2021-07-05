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



#ifndef CUCOMPAREUTILSCOMMON
#define CUCOMPAREUTILSCOMMON

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "MatrixAPI.hpp"
#include <math.h>

#define COMPARE_LIMIT 0.0001f

//__hostdevice__ floatt cuda_getRealDist(math::ComplexMatrix* m1, math::ComplexMatrix* m2, uint column, uint row) {
//  return cuda_getRealDist(m1, m2, column + gColumns (m1) * row);
//}

__hostdevice__ floatt cuda_getRealDist(math::ComplexMatrix* m1, math::ComplexMatrix* m2, uint index) {
  floatt re1 = gReValues (m1)[index];
  floatt im1 = gImValues (m1)[index];
  floatt re2 = gReValues (m2)[index];
  floatt im2 = gImValues (m2)[index];
  return sqrtf((re1-re2)*(re1-re2) - (im1-im2)*(im1-im2) - 2*re1*re2*im1*im2);
}

__hostdevice__ floatt cuda_getReDist(math::ComplexMatrix* m1, math::ComplexMatrix* m2, uint index) {
  return sqrtf((gReValues (m1)[index] - gReValues (m2)[index]) * (gReValues (m1)[index] - gReValues (m2)[index]));
}

__hostdevice__ floatt cuda_getImDist(math::ComplexMatrix* m1, math::ComplexMatrix* m2, uint index) {
  return sqrtf(-1 * (gImValues (m1)[index] - gImValues (m2)[index]) * (gImValues (m1)[index] - gImValues (m2)[index]));
}

__hostdevice__ int cuda_isEqualReIndex(math::ComplexMatrix* matrix1,
                                       math::ComplexMatrix* matrix2, uintt index) {
  HOST_INIT();
  return fabs(gReValues (matrix1)[index] - gReValues (matrix2)[index]) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualImIndex(math::ComplexMatrix* matrix1,
                                       math::ComplexMatrix* matrix2, uintt index) {
  HOST_INIT();
  return fabs(gImValues (matrix1)[index] - gImValues (matrix2)[index]) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualRe(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                                  uintt column, uintt row) {
  HOST_INIT();
  return fabs(GetRe(matrix1, column, row) - GetRe(matrix2, column, row)) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualIm(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                                  uintt column, uintt row) {
  HOST_INIT();
  return fabs(GetIm(matrix1, column, row) - GetIm(matrix2, column, row)) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

#endif  // CUCOMPAREUTILSCOMMON
