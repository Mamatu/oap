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

#ifndef OAP_API2_CU_TANH_PROCEDURES_H
#define OAP_API2_CU_TANH_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdeviceinline__ void cuda_ga_tanhFunc (floatt* output, floatt value)
{
  (*output) =  tanh (value);
}

__hostdeviceinline__ void cuda_ga_dtanhFunc (floatt* output, floatt value)
{
  floatt th = 0;
  cuda_tanhFunc (&th, value);
  (*output) =  (1.f  - th * th);
}

__hostdeviceinline__ void cuda_ga_mDTanhFunc (floatt* output, floatt value)
{
  floatt th = 0;
  cuda_tanhFunc (&th, value);
  (*output) =  (*output) * (1.f  - th * th);
}

__hostdeviceinline__ void cuda_genericApi_tanh (math::Matrix** outputs, math::Matrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_tanhFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_dtanh (math::Matrix** outputs, math::Matrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_dtanhFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDTanh (math::Matrix** outputs, math::Matrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_mDTanhFunc, mapper);
}

#endif
