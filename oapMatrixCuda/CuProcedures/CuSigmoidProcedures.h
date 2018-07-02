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

#ifndef CU_SIGMOID_PROCEDURES_H
#define CU_SIGMOID_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdeviceinline__ floatt sigmoid(floatt* output, floatt value)
{
  (*output) =  1.f / (1.f + exp(-value));
}

__hostdeviceinline__ void sigmoid2(floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void CUDA_sigmoidReMatrices(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoid(output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidImMatrices(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoid(output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidRealMatrices(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  sigmoid2(reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidMatrix(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_sigmoidRealMatrices(omatrix, imatrix);
  } else if (isre) {
    CUDA_sigmoidReMatrices(omatrix, imatrix);
  } else if (isim) {
    CUDA_sigmoidImMatrices(omatrix, imatrix);
  }
}

#endif /* CU_SIGMOID_PROCEDURES_H */
