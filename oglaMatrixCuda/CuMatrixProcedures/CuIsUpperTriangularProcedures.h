#ifndef CUISUPPERHESSENBERGPROCEDURES
#define CUISUPPERHESSENBERGPROCEDURES

#include "CuCore.h"
#include "Matrix.h"
#include "CuCompareUtils.h"

#define MIN_VALUE 0.001

/*@mamatu todo optimalization */
__hostdevice__ int CUDA_isUpperRealTriangular(math::Matrix* matrix) {
  HOST_INIT();
  uintt index = 0;
  int count = matrix->columns - 1;
  uintt columns = matrix->columns;
  for (uintt fa = 0; fa < columns - 1; ++fa) {
    floatt revalue = matrix->reValues[fa + columns * (fa + 1)];
    floatt imvalue = matrix->imValues[fa + columns * (fa + 1)];
    if ((-MIN_VALUE < revalue && revalue < MIN_VALUE) &&
        (-MIN_VALUE < imvalue && imvalue < MIN_VALUE)) {
      ++index;
    }
  }
  return index >= count;
}

/*@mamatu todo optimalization */
__hostdevice__ int CUDA_isUpperReTriangular(math::Matrix* matrix) {
  HOST_INIT();
  uintt index = 0;
  int count = matrix->columns - 1;
  uintt columns = matrix->columns;
  for (uintt fa = 0; fa < columns - 1; ++fa) {
    floatt revalue = matrix->reValues[fa + columns * (fa + 1)];
    if (-MIN_VALUE < revalue && revalue < MIN_VALUE) {
      ++index;
    }
  }
  return index >= count;
}

/*@mamatu todo optimalization */
__hostdevice__ int CUDA_isUpperImTriangular(math::Matrix* matrix) {
  HOST_INIT();
  uintt index = 0;
  int count = matrix->columns - 1;
  uintt columns = matrix->columns;
  for (uintt fa = 0; fa < columns - 1; ++fa) {
    floatt imvalue = matrix->imValues[fa + columns * (fa + 1)];
    if (-MIN_VALUE < imvalue && imvalue < MIN_VALUE) {
      ++index;
    }
  }
  return index >= count;
}

__hostdevice__ int CUDA_isUpperTriangular(math::Matrix* matrix) {
  HOST_INIT();
  bool isre = matrix->reValues != NULL;
  bool isim = matrix->imValues != NULL;
  if (isre && isim) {
    return CUDA_isUpperRealTriangular(matrix);
  } else if (isre) {
    return CUDA_isUpperReTriangular(matrix);
  } else if (isim) {
    return CUDA_isUpperImTriangular(matrix);
  }
}

#endif  // CUISTRIANGULARPROCEDURES
