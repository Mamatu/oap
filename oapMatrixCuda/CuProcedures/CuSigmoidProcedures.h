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

#ifndef CU_SIGMOID_PROCEDURES_H
#define CU_SIGMOID_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdeviceinline__ void cuda_sigmoidFunc (floatt* output, floatt value)
{
  (*output) =  (1. / (1. + exp(-value)));
}

__hostdeviceinline__ void cuda_dsigmoidFunc (floatt* output, floatt value)
{
  floatt sv = 0;
  cuda_sigmoidFunc (&sv, value);
  (*output) =  sv * (1.f  - sv);
}

__hostdeviceinline__ void cuda_mDSigmoidFunc (floatt* output, floatt value)
{
  floatt sv = 0;
  cuda_sigmoidFunc (&sv, value);
  (*output) =  (*output) * sv * (1.f  - sv);
}

__hostdeviceinline__ void cuda_sigmoid (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_sigmoidFunc);
}

__hostdeviceinline__ void cuda_dsigmoid (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_dsigmoidFunc);
}

__hostdeviceinline__ void cuda_multiplyDSigmoid (math::Matrix* output, math::Matrix* matrix)
{
  cuda_func (output, matrix, cuda_mDSigmoidFunc);
}

#endif
