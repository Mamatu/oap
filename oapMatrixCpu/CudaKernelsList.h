/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef OAP_CUDA_KERNELS_LIST_H
#define OAP_CUDA_KERNELS_LIST_H

#include "CuMatrixProcedures.h"

void HOSTKernel_SumShared (floatt* output[2], math::Matrix* params0)
{
  CUDA_sumShared (output, params0);
}

void HOSTKernel_SumSharedRaw (void** params)
{
  floatt** param1 = static_cast<floatt**> (params[0]);
  math::Matrix* param2 = static_cast<math::Matrix*> (params[1]);
  HOSTKernel_SumShared (param1, param2);
}


#endif


