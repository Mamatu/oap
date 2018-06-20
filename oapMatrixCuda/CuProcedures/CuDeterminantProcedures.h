/*
 * Copyright 2016 - 2018 Marcin Matula
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
