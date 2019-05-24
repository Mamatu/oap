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

#ifndef CU_SIN_PROCEDURES_H
#define CU_SIN_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdeviceinline__ void sinReal (floatt* output, floatt value)
{
  (*output) =  sin (value);
}

__hostdeviceinline__ void sinDerivative (floatt* output, floatt value)
{
  (*output) =  cos (value);
}

__hostdeviceinline__ void multiplySinDerivative (floatt* output, floatt value)
{
  (*output) =  (*output) * cos (value);
}

__hostdeviceinline__ void multiplySinComplexDerivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sinComplexDerivative (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sinComplex (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void CUDA_sinRe(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sinReal (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sinIm(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  sinReal (output, imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sinReal(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  sinComplex (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sin (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_sinReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_sinRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_sinIm (omatrix, imatrix);
  }
}

__hostdeviceinline__ void CUDA_sinDerivativeRe (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  sinDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sinDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->imValues[index];
  sinDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sinDerivativeReal (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->imValues[index];
  sinComplexDerivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_sinDerivative(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_sinDerivativeReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_sinDerivativeRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_sinDerivativeIm (omatrix, imatrix);
  }
}

__hostdeviceinline__ void CUDA_multiplySinDerivativeRe (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplySinDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySinDerivativeIm (math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* output = &omatrix->reValues[index];
  multiplySinDerivative (output, imatrix->reValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySinDerivativeReal(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();

  uintt offset = omatrix->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;

  floatt* reoutput = &omatrix->reValues[index];
  floatt* imoutput = &omatrix->reValues[index];
  multiplySinComplexDerivative (reoutput, imoutput, imatrix->reValues[index], imatrix->imValues[index]);

  threads_sync();
}

__hostdeviceinline__ void CUDA_multiplySinDerivative(math::Matrix* omatrix, math::Matrix* imatrix)
{
  HOST_INIT();
  bool isre = omatrix->reValues != NULL;
  bool isim = omatrix->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplySinDerivativeReal (omatrix, imatrix);
  } else if (isre) {
    CUDA_multiplySinDerivativeRe (omatrix, imatrix);
  } else if (isim) {
    CUDA_multiplySinDerivativeIm (omatrix, imatrix);
  }
}

#endif /* CU_SIN_PROCEDURES_H */
