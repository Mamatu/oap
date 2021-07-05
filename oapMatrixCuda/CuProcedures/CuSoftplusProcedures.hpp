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

#ifndef CU_SOFTPLUS_PROCEDURES_H
#define CU_SOFTPLUS_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "CuFuncProcedures.hpp"

__hostdeviceinline__ void cuda_softplusFunc (floatt* output, floatt value)
{
  (*output) =  logf (1. + expf (value));
}

__hostdeviceinline__ void cuda_dsoftplusFunc (floatt* output, floatt value)
{
  (*output) =  1. / (1. + expf (-value));
}

__hostdeviceinline__ void cuda_softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  cuda_func (output, matrix, cuda_softplusFunc);
}

__hostdeviceinline__ void cuda_dsoftplus(math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  cuda_func (output, matrix, cuda_dsoftplusFunc);
}

#endif /* CU_SOFTPLUS_PROCEDURES_H */
