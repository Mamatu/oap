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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_API2_CU_TENSOR_PRODUCT_PROCEDURES_H
#define OAP_API2_CU_TENSOR_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "oapMemory_ThreadMapperApi.h"
#include "oapThreadsMapperS.h"

#include "../CuCreateProcedures.h"
#include "../CuDotProductSpecificProcedures.h"

__hostdevice__ void cuda_GenericApi_tensorProductRe (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);
 
  if (inrange)
  {
    math::ComplexMatrix* output = outputs[oidxs[0]];
    math::ComplexMatrix* param1 = params0[oidxs[0]];
    math::ComplexMatrix* param2 = params1[oidxs[0]];
   
    const uintt columns1 = GetColumns (param1);
    const uintt columns2 = GetColumns (param2);
    const uintt offset = columns1;
  
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];
  
    uintt params1_index_y = y % gRows (param1);
    uintt params0_section_y = y / gRows (param1);
  
    uintt params1_index_x = x % gColumns (param1);
    uintt params0_section_x = x / gColumns (param1);
  
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re.mem, param1->re.reg, params0_section_x, params0_section_y);
    uintt idx2 = oap::common::GetMemIdxFromMatrixPos (param2->re.mem, param2->re.reg, params1_index_x, params1_index_y);
  
    uintt index = oap::common::GetMemIdxFromMatrixPos (output->re.mem, output->re.reg, x, y);
    output->re.mem.ptr[index] = param1->re.mem.ptr[idx1] * param2->re.mem.ptr[idx2];
  }
}

__hostdevice__ void cuda_GenericApi_tensorProductIm (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  assert ("Not supported" != NULL);
}

__hostdevice__ void cuda_GenericApi_tensorProductReal (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  assert ("Not supported" != NULL);
}

__hostdevice__ void CUDA_GenericApi_tensorProductRe (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
 
  cuda_GenericApi_tensorProductRe(output, params0, params1, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_tensorProductIm (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
 
  cuda_GenericApi_tensorProductIm(output, params0, params1, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_tensorProductReal (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
 
  cuda_GenericApi_tensorProductReal(output, params0, params1, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_TensorProduct (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  bool isRe = output[0]->re.mem.ptr != NULL;
  bool isIm = output[0]->im.mem.ptr != NULL;
 
  if (isRe && isIm)
  {
    CUDA_GenericApi_tensorProductReal (output, params0, params1, mapper);
  }
  else if (isRe)
  {
    CUDA_GenericApi_tensorProductRe (output, params0, params1, mapper);
  }
  else if (isIm)
  {
    CUDA_GenericApi_tensorProductIm (output, params0, params1, mapper);
  }
}

#endif
