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

#ifndef OAP_API2_CU_DOT_PRODUCT_SHARED_PROCEDURES_H
#define OAP_API2_CU_DOT_PRODUCT_SHARED_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "oapMemory_ThreadMapperApi.h"
#include "oapThreadsMapperS.h"

#include "../CuCreateProcedures.h"
#include "../CuDotProductSpecificProcedures.h"

__hostdevice__ void cuda_GenericApi_dotProductSharedRe (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
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
    floatt retemp = 0;

    for (uintt idx = 0; idx < offset; ++idx)
    {
      uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re.mem, param1->re.reg, idx, y);
      uintt idx2 = oap::common::GetMemIdxFromMatrixPos (param2->re.mem, param2->re.reg, x, idx);
      retemp += param1->re.mem.ptr[idx1] * param2->re.mem.ptr[idx2];
    }

    uintt index = oap::common::GetMemIdxFromMatrixPos (output->re.mem, output->re.reg, x, y);
    output->re.mem.ptr[index] = retemp;
  }
}

__hostdevice__ void cuda_GenericApi_dotProductSharedIm (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
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
    floatt imtemp = 0;

    for (uintt idx = 0; idx < offset; ++idx)
    {
      uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->im.mem, param1->im.reg, idx, y);
      uintt idx2 = oap::common::GetMemIdxFromMatrixPos (param2->im.mem, param2->im.reg, x, idx);
      imtemp += param1->im.mem.ptr[idx1] * param2->im.mem.ptr[idx2] * -1.;
    }

    uintt index = oap::common::GetMemIdxFromMatrixPos (output->im.mem, output->im.reg, x, y);
    output->im.mem.ptr[index] = imtemp;
  }
}

__hostdevice__ void cuda_GenericApi_dotProductSharedReal (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
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
    floatt retemp = 0;
    floatt imtemp = 0;

    for (uintt idx = 0; idx < offset; ++idx)
    {
      {
        uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re.mem, param1->re.reg, idx, y);
        uintt idx2 = oap::common::GetMemIdxFromMatrixPos (param2->re.mem, param2->re.reg, x, idx);
        retemp += param1->re.mem.ptr[idx1] * param2->re.mem.ptr[idx2];
      }
      {
        uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->im.mem, param1->im.reg, idx, y);
        uintt idx2 = oap::common::GetMemIdxFromMatrixPos (param2->im.mem, param2->im.reg, x, idx);
        imtemp += param1->im.mem.ptr[idx1] * param2->im.mem.ptr[idx2] * -1.;
      }
    }

    {
      uintt index = oap::common::GetMemIdxFromMatrixPos (output->re.mem, output->re.reg, x, y);
      output->re.mem.ptr[index] = retemp;
    }
    {
      uintt index = oap::common::GetMemIdxFromMatrixPos (output->im.mem, output->im.reg, x, y);
      output->im.mem.ptr[index] = imtemp;
    }
  }
}

__hostdevice__ void CUDA_GenericApi_dotProductSharedRe (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  cuda_GenericApi_dotProductSharedRe(output, params0, params1, buffer, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProductSharedIm (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  cuda_GenericApi_dotProductSharedIm(output, params0, params1, buffer, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProductSharedReal (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  cuda_GenericApi_dotProductSharedReal(output, params0, params1, buffer, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_DotProduct (math::ComplexMatrix** output, math::ComplexMatrix* const* params0, math::ComplexMatrix* const* params1, floatt* buffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isRe = output[0]->re.mem.ptr != NULL;
  bool isIm = output[0]->im.mem.ptr != NULL;

  if (isRe && isIm)
  {
    CUDA_GenericApi_dotProductSharedReal (output, params0, params1, buffer, mapper);
  }
  else if (isRe)
  {
    CUDA_GenericApi_dotProductSharedRe (output, params0, params1, buffer, mapper);
  }
  else if (isIm)
  {
    CUDA_GenericApi_dotProductSharedIm (output, params0, params1, buffer, mapper);
  }
}

#endif
