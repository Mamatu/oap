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




#ifndef CUIDENTITYPROCEDURES_H
#define CUIDENTITYPROCEDURES_H

#include "CuCore.h"
#include "MatrixAPI.h"

__hostdevice__ void CUDA_SetIdentityReMatrix(math::Matrix* dst) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt v = threadIndexX == threadIndexY ? 1 : 0;
  SetRe(dst, threadIndexX, threadIndexY, v);
  threads_sync();
}

__hostdevice__ void CUDA_SetIdentityImMatrix(math::Matrix* dst) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt v = threadIndexX == threadIndexY ? 1 : 0;
  SetIm(dst, threadIndexX, threadIndexY, v);
  threads_sync();
}

__hostdevice__ void CUDA_SetIdentityMatrix(math::Matrix* dst) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt v = threadIndexX == threadIndexY ? 1 : 0;
  SetRe(dst, threadIndexX, threadIndexY, v);
  if (dst->im.ptr != NULL) {
    SetIm(dst, threadIndexX, threadIndexY, 0);
  }
  threads_sync();
}

#endif /* CUIDENTITYPROCEDURES_H */
