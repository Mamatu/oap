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
  (*output) =  (1. / (1. + exp(-value)));
}

__hostdeviceinline__ void sigmoidDerivative (floatt* output, floatt value)
{
  floatt sv = 0;
  sigmoid(&sv, value);
  (*output) =  sv * (1.f  - sv);
}

__hostdeviceinline__ void multiplySigmoidDerivative (floatt* output, floatt value)
{
  floatt sv = 0;
  sigmoid(&sv, value);
  (*output) =  (*output) * sv * (1.f  - sv);
}

__hostdeviceinline__ void multiplySigmoidComplexDerivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sigmoidComplexDerivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sigmoidComplex (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void CUDA_sigmoidRe(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoid (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidIm(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  sigmoid(output, imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidReal(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  sigmoidComplex (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoid (math::Matrix* omatrix, math::Matrix* imatrix)
{
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

__hostdeviceinline__ void CUDA_sigmoidDerivativeRe (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  sigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidDerivativeReal (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  sigmoidComplexDerivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sigmoidDerivative(math::Matrix* omatrix, math::Matrix* imatrix)
{
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

__hostdeviceinline__ void CUDA_multiplySigmoidDerivativeRe (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplySigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplySigmoidDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivativeReal(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  multiplySigmoidComplexDerivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySigmoidDerivative(math::Matrix* omatrix, math::Matrix* imatrix)
{
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
