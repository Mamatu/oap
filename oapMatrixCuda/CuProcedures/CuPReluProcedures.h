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

#ifndef CU_PRELU_PROCEDURES_H
#define CU_PRELU_PROCEDURES_H

#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdevice__ void cuda_preluFunc (floatt* output, floatt value, void* ud)
{
  floatt alpha = *(floatt*) ud;
  (*output) =  (value > 0.) ? value : (alpha * value);
}

__hostdevice__ void cuda_dpreluFunc (floatt* output, floatt value, void* ud)
{
  floatt alpha = *(floatt*) ud;
  (*output) = (value > 0.) ? 1. : alpha;
  //cuda_debug ("value = %f", value);
}

__hostdeviceinline__ void cuda_prelu_alpha (math::Matrix* output, math::Matrix* matrix, floatt alpha)
{
  cuda_func_userData (output, matrix, cuda_preluFunc, &alpha);
}

__hostdeviceinline__ void cuda_dprelu_alpha (math::Matrix* output, math::Matrix* matrix, floatt alpha)
{
  cuda_func_userData (output, matrix, cuda_dpreluFunc, &alpha);
}

__hostdeviceinline__ void cuda_prelu (math::Matrix* output, math::Matrix* matrix)
{
  cuda_prelu_alpha (output, matrix, 0.01);
}

__hostdeviceinline__ void cuda_dprelu (math::Matrix* output, math::Matrix* matrix)
{
  cuda_dprelu_alpha (output, matrix, 0.01);
}

#endif /* CU_PRELU_PROCEDURES_H */
