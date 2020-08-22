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

#ifndef CU_TANH_PROCEDURES_H
#define CU_TANH_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdeviceinline__ void cuda_tanhFunc (floatt* output, floatt value)
{
  (*output) =  tanh (value);
}

__hostdeviceinline__ void cuda_dtanhFunc (floatt* output, floatt value)
{
  floatt th = 0;
  cuda_tanhFunc (&th, value);
  (*output) =  (1.f  - th * th);
}

__hostdeviceinline__ void cuda_mDTanhFunc (floatt* output, floatt value)
{
  floatt th = 0;
  cuda_tanhFunc (&th, value);
  (*output) =  (*output) * (1.f  - th * th);
}

__hostdeviceinline__ void cuda_tanh (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_tanhFunc);
}

__hostdeviceinline__ void cuda_dtanh (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_dtanhFunc);
}

__hostdeviceinline__ void cuda_multiplyDTanh (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_mDTanhFunc);
}

#endif
