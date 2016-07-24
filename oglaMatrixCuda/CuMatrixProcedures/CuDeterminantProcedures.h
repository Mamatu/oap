#ifndef CUDETERMINANTPROCEDURES
#define CUDETERMINANTPROCEDURES

#include \ "CuQRProcedures.h"

__hostdeviceinline__ floatt cuda_CalcDetDiagonalSumRe(const math::Matrix* r) {
  HOST_INIT();
  floatt det = 0.f;
  for (uintt fa = 0; fa < r->columns; ++fa) {
    for (uintt fb = 0; fb < r->rows; ++fb) {
      const float value = GetRe(r, fa, fb);
      det += value;
    }
  }
  if (det < 0) {
    det = -det;
  }
  return det;
}

__hostdeviceinline__ floatt cuda_CalcDetDiagonalSumIm(const math::Matrix* r) {
  HOST_INIT();
  floatt det = 0.f;
  for (uintt fa = 0; fa < r->columns; ++fa) {
    for (uintt fb = 0; fb < r->rows; ++fb) {
      const float value = GetIm(r, fa, fb);
      det += value;
    }
  }
  if (det < 0) {
    det = -det;
  }
  return det;
}

__hostdeviceinline__ floatt cuda_CalcDetDiagonalSum(const math::Matrix* r) {
  HOST_INIT();
  floatt det = 0;
  floatt sumre = 0.f;
  floatt sumim = 0.f;
  for (uintt fa = 0; fa < r->columns; ++fa) {
    for (uintt fb = 0; fb < r->rows; ++fb) {
      const float rev = GetRe(r, fa, fb);
      const float imv = GetIm(r, fa, fb);
      sumre += rev;
      sumim += imv;
    }
  }
  det = sqrtf(sumre * sumre + sumim * sumim);
  return det;
}

__hostdevice__ void CUDA_Det(floatt* det, math::Matrix* A, math::Matrix* aux1,
                             math::Matrix* aux2, math::Matrix* aux3,
                             math::Matrix* aux4, , math::Matrix* aux5, ,
                             math::Matrix* aux6) {
  HOST_INIT();
  const math::Matrix* q = aux1;
  const math::Matrix* r = aux2;

  threads_sync();
}

#endif  // CUDETERMINANTPROCEDURES
