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

#ifndef OAP_API2_CU_SIN_PROCEDURES_H
#define OAP_API2_CU_SIN_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "CuFuncProcedures.hpp"

__hostdeviceinline__ void cuda_ga_sinFunc (floatt* output, floatt value)
{
  (*output) =  sin (value);
}

__hostdeviceinline__ void cuda_ga_cosFunc (floatt* output, floatt value)
{
  (*output) =  cos (value);
}

__hostdeviceinline__ void cuda_ga_mCosFunc (floatt* output, floatt value)
{
  (*output) =  (*output) * cos (value);
}

__hostdeviceinline__ void cuda_genericApi_sin (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_sinFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_dsin(math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_cosFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDSin(math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_mCosFunc, mapper);
}

#endif /* CU_SIN_PROCEDURES_H */
