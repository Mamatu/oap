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

#ifndef CU_SIN_PROCEDURES_H
#define CU_SIN_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdeviceinline__ void cuda_sinFunc (floatt* output, floatt value)
{
  (*output) =  sin (value);
}

__hostdeviceinline__ void cuda_cosFunc (floatt* output, floatt value)
{
  (*output) =  cos (value);
}

__hostdeviceinline__ void cuda_mCosFunc (floatt* output, floatt value)
{
  (*output) =  (*output) * cos (value);
}

__hostdeviceinline__ void multiplyDSinComplex (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void dsinComplex (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void sinComplex (floatt* reoutput, floatt* imoutput, floatt revalue, floatt imvalue)
{
}

__hostdeviceinline__ void cuda_sin (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_sinFunc);
}

__hostdeviceinline__ void cuda_dsin(math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_cosFunc);
}

__hostdeviceinline__ void cuda_multiplyDSin(math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_mCosFunc);
}

#endif /* CU_SIN_PROCEDURES_H */
