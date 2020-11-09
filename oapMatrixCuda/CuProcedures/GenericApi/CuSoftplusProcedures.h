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

#ifndef OAP_API2_CU_SOFTPLUS_PROCEDURES_H
#define OAP_API2_CU_SOFTPLUS_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuFuncProcedures.h"

__hostdeviceinline__ void cuda_genericApi_softplusFunc (floatt* output, floatt value)
{
  (*output) =  logf (1. + expf (value));
}

__hostdeviceinline__ void cuda_genericApi_dsoftplusFunc (floatt* output, floatt value)
{
  (*output) =  1. / (1. + expf (-value));
}

__hostdeviceinline__ void cuda_genericApi_multiplyDSoftplusFunc (floatt* output, floatt value)
{
  (*output) =  (*output) / (1. + expf (-value));
}

__hostdeviceinline__ void cuda_genericApi_softplus (math::Matrix** outputs, math::Matrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_genericApi_softplusFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_dsoftplus (math::Matrix** outputs, math::Matrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_genericApi_dsoftplusFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDSoftplus (math::Matrix** outputs, math::Matrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_genericApi_multiplyDSoftplusFunc, mapper);
}

#endif /* CU_SOFTPLUS_PROCEDURES_H */
