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

#ifndef OAP_API2_CU_RELU_PROCEDURES_H
#define OAP_API2_CU_RELU_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "CuPReluProcedures.hpp"

__hostdeviceinline__ void cuda_genericApi_relu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  floatt alpha = 0.;
  cuda_func_userData (outputs, params, cuda_genericApi_preluFunc, &alpha, mapper);
}

__hostdeviceinline__ void cuda_genericApi_drelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  floatt alpha = 0.;
  cuda_func_userData (outputs, params, cuda_genericApi_dpreluFunc, &alpha, mapper);
}

__hostdeviceinline__ void cuda_genericApi_multiplyDRelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  floatt alpha = 0.;
  cuda_func_userData (outputs, params, cuda_genericApi_multiplyDPreluFunc, &alpha, mapper);
}

#endif /* CU_PRELU_PROCEDURES_H */
