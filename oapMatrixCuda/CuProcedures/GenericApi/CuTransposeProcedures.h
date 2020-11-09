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

#ifndef OAP_API2_CU_TRANSPOSE_PROCEDURES_H
#define OAP_API2_CU_TRANSPOSE_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "oapMemory_ThreadMapperApi.h"
#include "oapThreadsMapperS.h"

#include "../CuCreateProcedures.h"
#include "../CuDotProductSpecificProcedures.h"

__hostdevice__ void cuda_GenericApi_transposeRe (math::Matrix** outputs, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);
 
  if (inrange)
  {
    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param1 = params0[oidxs[0]];
   
    const uintt columns1 = GetColumns (param1);
    const uintt offset = columns1;
  
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];
  
    uintt index = oap::common::GetMemIdxFromMatrixPos (output->re, output->reReg, x, y);
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re, param1->reReg, y, x);
  
    output->re.ptr[index] = param1->re.ptr[idx1];
  }
}

__hostdevice__ void cuda_GenericApi_transposeIm (math::Matrix** outputs, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);
 
  if (inrange)
  {
    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param1 = params0[oidxs[0]];
   
    const uintt columns1 = GetColumns (param1);
    const uintt offset = columns1;
  
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];
  
    uintt index = oap::common::GetMemIdxFromMatrixPos (output->im, output->imReg, x, y);
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->im, param1->imReg, y, x);
  
    output->re.ptr[index] = param1->re.ptr[idx1];
  }
}

__hostdevice__ void cuda_GenericApi_transposeReal (math::Matrix** outputs, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);
 
  if (inrange)
  {
    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param1 = params0[oidxs[0]];
   
    const uintt columns1 = GetColumns (param1);
    const uintt offset = columns1;
  
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];
  
    {
    uintt index = oap::common::GetMemIdxFromMatrixPos (output->re, output->reReg, x, y);
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re, param1->reReg, y, x);
  
    output->re.ptr[index] = param1->re.ptr[idx1];
    }
    {
    uintt index = oap::common::GetMemIdxFromMatrixPos (output->im, output->imReg, x, y);
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->im, param1->imReg, y, x);
  
    output->re.ptr[index] = param1->re.ptr[idx1];
    }
  }
}

__hostdevice__ void CUDA_GenericApi_transposeRe (math::Matrix** output, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
 
  cuda_GenericApi_transposeRe(output, params0, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_transposeIm (math::Matrix** output, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
 
  cuda_GenericApi_transposeIm(output, params0, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_transposeReal (math::Matrix** output, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
 
  cuda_GenericApi_transposeReal(output, params0, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_Transpose (math::Matrix** output, math::Matrix* const* params0, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  bool isRe = output[0]->re.ptr != NULL;
  bool isIm = output[0]->im.ptr != NULL;
 
  if (isRe && isIm)
  {
    CUDA_GenericApi_transposeReal (output, params0, mapper);
  }
  else if (isRe)
  {
    CUDA_GenericApi_transposeRe (output, params0, mapper);
  }
  else if (isIm)
  {
    CUDA_GenericApi_transposeIm (output, params0, mapper);
  }
}

#endif
