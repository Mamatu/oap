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

#ifndef OAP_API2_CU_PRELU_PROCEDURES_H
#define OAP_API2_CU_PRELU_PROCEDURES_H

#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdevice__ void cuda_genericApi_preluFunc (floatt* output, floatt value, void* ud)
{
  floatt alpha = *(floatt*) ud;
  (*output) =  (value > 0.) ? value : (alpha * value);
}

__hostdevice__ void cuda_genericApi_dpreluFunc (floatt* output, floatt value, void* ud)
{
  floatt alpha = *(floatt*) ud;
  (*output) = (value > 0.) ? 1. : alpha;
}

__hostdevice__ void cuda_genericApi_multiplyDPreluFunc (floatt* output, floatt value, void* ud)
{
  floatt alpha = *(floatt*) ud;
  (*output) = (*output) * ((value > 0.) ? 1. : alpha);
}

__hostdeviceinline__ void cuda_genericApi_prelu_alpha (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, floatt alpha, oap::ThreadsMapperS* mapper)
{
  cuda_func_userData (outputs, params, cuda_genericApi_preluFunc, &alpha, mapper);
}

__hostdeviceinline__ void cuda_genericApi_dprelu_alpha (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, floatt alpha, oap::ThreadsMapperS* mapper)
{
  cuda_func_userData (outputs, params, cuda_genericApi_dpreluFunc, &alpha, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDPrelu_alpha (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, floatt alpha, oap::ThreadsMapperS* mapper)
{
  cuda_func_userData (outputs, params, cuda_genericApi_multiplyDPreluFunc, &alpha, mapper);
}

__hostdeviceinline__ void cuda_genericApi_prelu (math::ComplexMatrix** output, math::ComplexMatrix* const* matrix, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_prelu_alpha (output, matrix, 0.01, mapper);
}

__hostdeviceinline__ void cuda_genericApi_dprelu (math::ComplexMatrix** output, math::ComplexMatrix* const* matrix, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_dprelu_alpha (output, matrix, 0.01, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDPrelu (math::ComplexMatrix** output, math::ComplexMatrix* const* matrix, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_multiplyDPrelu_alpha (output, matrix, 0.01, mapper);
}

#endif /* CU_PRELU_PROCEDURES_H */
