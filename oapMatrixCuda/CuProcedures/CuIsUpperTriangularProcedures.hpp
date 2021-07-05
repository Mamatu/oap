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



#ifndef CUISUPPERHESSENBERGPROCEDURES
#define CUISUPPERHESSENBERGPROCEDURES

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "CuCompareUtils.hpp"

#define MIN_VALUE 0.001

/*@mamatu todo optimalization */
__hostdevice__ int CUDA_isUpperRealTriangular(math::ComplexMatrix* matrix) {
  HOST_INIT();
  uintt index = 0;
  int count = gColumns (matrix) - 1;
  uintt columns = gColumns (matrix);
  for (uintt fa = 0; fa < columns - 1; ++fa) {
    floatt revalue = *GetRePtrIndex (matrix, fa + columns * (fa + 1));
    floatt imvalue = *GetImPtrIndex (matrix, fa + columns * (fa + 1));
    if ((-MIN_VALUE < revalue && revalue < MIN_VALUE) &&
        (-MIN_VALUE < imvalue && imvalue < MIN_VALUE)) {
      ++index;
    }
  }
  return index == count;
}

/*@mamatu todo optimalization */
__hostdevice__ int CUDA_isUpperReTriangular(math::ComplexMatrix* matrix) {
  HOST_INIT();
  uintt index = 0;
  int count = gColumns (matrix) - 1;
  uintt columns = gColumns (matrix);
  for (uintt fa = 0; fa < columns - 1; ++fa) {
    floatt revalue = *GetRePtrIndex (matrix, fa + columns * (fa + 1));
    if (-MIN_VALUE < revalue && revalue < MIN_VALUE) {
      ++index;
    }
  }
  return index == count;
}

/*@mamatu todo optimalization */
__hostdevice__ int CUDA_isUpperImTriangular(math::ComplexMatrix* matrix) {
  HOST_INIT();
  uintt index = 0;
  int count = gColumns (matrix) - 1;
  uintt columns = gColumns (matrix);
  for (uintt fa = 0; fa < columns - 1; ++fa) {
    floatt imvalue = *GetImPtrIndex (matrix, fa + columns * (fa + 1));
    if (-MIN_VALUE < imvalue && imvalue < MIN_VALUE) {
      ++index;
    }
  }
  return index == count;
}

__hostdevice__ int CUDA_isUpperTriangular(math::ComplexMatrix* matrix) {
  HOST_INIT();
  bool isre = matrix->re.mem.ptr != NULL;
  bool isim = matrix->im.mem.ptr != NULL;
  if (isre && isim) {
    return CUDA_isUpperRealTriangular(matrix);
  } else if (isre) {
    return CUDA_isUpperReTriangular(matrix);
  } else if (isim) {
    return CUDA_isUpperImTriangular(matrix);
  }
}

#endif  // CUISTRIANGULARPROCEDURES
