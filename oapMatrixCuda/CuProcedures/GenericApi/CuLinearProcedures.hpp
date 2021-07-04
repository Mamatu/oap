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

#ifndef OAP_API2_CU_LINEAR_PROCEDURES_H
#define OAP_API2_CU_LINEAR_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "CuFuncProcedures.hpp"

__hostdeviceinline__ void cuda_ga_linearFunc (floatt* output, floatt value)
{
  (*output) =  value;
}

__hostdeviceinline__ void cuda_ga_dlinearFunc (floatt* output, floatt value)
{
  (*output) =  1;
}

__hostdeviceinline__ void cuda_ga_mLinearFunc (floatt* output, floatt value)
{
  (*output) =  (*output) * 1;
}

__hostdeviceinline__ void cuda_genericApi_linear (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_linearFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_dlinear (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_dlinearFunc, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDLinear (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_func (outputs, params, cuda_ga_mLinearFunc, mapper);
}

#endif /* CU_SIN_PROCEDURES_H */
