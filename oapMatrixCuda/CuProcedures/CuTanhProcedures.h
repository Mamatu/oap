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

#ifndef CU_TANH_PROCEDURES_H
#define CU_TANH_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdeviceinline__ void tanhFunc (floatt* output, floatt value)
{
  (*output) =  tanh (value);
}

__hostdeviceinline__ void tanhDerivative (floatt* output, floatt value)
{
  floatt th = 0;
  tanhFunc(&th, value);
  (*output) =  (1.f  - th * th);
}

__hostdeviceinline__ void multiplyTanhDerivative (floatt* output, floatt value)
{
  floatt th = 0;
  tanhFunc(&th, value);
  (*output) =  (*output) * (1.f  - th * th);
}

__hostdeviceinline__ void multiplyTanhComplexDerivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void tanhComplexDerivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void tanhComplex (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void CUDA_tanhRe(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  tanhFunc (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_tanhIm(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  tanhFunc (output, imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_tanhReal(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  tanhComplex (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_tanh (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_tanhReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_tanhRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_tanhIm (omatrix, imatrix);
  }
}

__hostdeviceinline__ void CUDA_tanhDerivativeRe (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  tanhDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_tanhDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  tanhDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_tanhDerivativeReal (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  tanhComplexDerivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_tanhDerivative(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_tanhDerivativeReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_tanhDerivativeRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_tanhDerivativeIm (omatrix, imatrix);
  }
}

__hostdeviceinline__ void CUDA_multiplyTanhDerivativeRe (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplyTanhDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplyTanhDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplyTanhDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplyTanhDerivativeReal(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  multiplyTanhComplexDerivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplyTanhDerivative(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplyTanhDerivativeReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_multiplyTanhDerivativeRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_multiplyTanhDerivativeIm (omatrix, imatrix);
  }
}

#endif /* CU_TANH_PROCEDURES_H */
