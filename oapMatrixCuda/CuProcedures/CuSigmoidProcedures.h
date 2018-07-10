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

__hostdeviceinline__ void sigmoid (floatt* output, floatt value)
{
  (*output) =  1.f / (1.f + exp(-value));
}

__hostdeviceinline__ void sigmoidDerivative (floatt* output, floatt value)
{
  (*output) =  1.f * (1.f  - value);
}

__hostdeviceinline__ void multiplySigmoidDerivative (floatt* output, floatt value)
{
  (*output) =  (*output) * (1.f  - value);
}

__hostdeviceinline__ void multiplySigmoid2Derivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sigmoid2Derivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sigmoid2(floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void CUDA_sigmoidRe(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoid(output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidIm(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoid(output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidReal(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  sigmoid2(reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoid(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_sigmoidReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_sigmoidRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_sigmoidIm (omatrix, imatrix);
  }
}

__hostdeviceinline__ void CUDA_sigmoidDerivativeRe(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidDerivativeReal(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  sigmoid2Derivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidDerivative(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_sigmoidDerivativeReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_sigmoidDerivativeRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_sigmoidDerivativeIm (omatrix, imatrix);
  }
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivativeRe(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplySigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplySigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivativeReal(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  multiplySigmoid2Derivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivative(math::Matrix* omatrix, math::Matrix* imatrix) {
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplySigmoidDerivativeReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_multiplySigmoidDerivativeRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_multiplySigmoidDerivativeIm (omatrix, imatrix);
  }
}

#endif /* CU_SIGMOID_PROCEDURES_H */
