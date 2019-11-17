/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef CU_RELU_PROCEDURES_H
#define CU_RELU_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdeviceinline__ void cuda_reluFunc (floatt* output, floatt value)
{
  (*output) =  value > 0. ? value : 0.;
}

__hostdeviceinline__ void cuda_dreluFunc (floatt* output, floatt value)
{
  (*output) =  value > 0. ? 1. : 0.;
}

__hostdeviceinline__ void CUDA_relu (math::Matrix* output, math::Matrix* matrix)
{
  CUDA_func (output, matrix, cuda_reluFunc);
}

__hostdeviceinline__ void CUDA_drelu(math::Matrix* output, math::Matrix* matrix)
{
  CUDA_func (output, matrix, cuda_dreluFunc);
}

#endif /* CU_RELU_PROCEDURES_H */
