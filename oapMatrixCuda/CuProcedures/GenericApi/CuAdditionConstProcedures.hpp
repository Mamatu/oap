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

#ifndef OAP_API2_CU_ADDITION_CONST_PROCEDURES_H
#define OAP_API2_CU_ADDITION_CONST_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "MatrixAPI.hpp"
#include "oapMemory_ThreadMapperApi.hpp"
#include "oapThreadsMapperS.hpp"

__hostdeviceinline__ void cuda_GenericApi_addReConst (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[3];

  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    math::ComplexMatrix* output = outputs[oidxs[0]];
    const math::ComplexMatrix* param1 = params1[oidxs[0]];

    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    uintt index = oap::common::GetMemIdxFromMatrixPos (output->re.mem, output->re.reg, x, y);
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re.mem, param1->re.reg, x, y);

    output->re.mem.ptr[index] = param1->re.mem.ptr[idx1] + value;
  }
}

__hostdeviceinline__ void cuda_GenericApi_addImConst (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[3];

  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    math::ComplexMatrix* output = outputs[oidxs[0]];
    const math::ComplexMatrix* param1 = params1[oidxs[0]];

    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    uintt index = oap::common::GetMemIdxFromMatrixPos (output->im.mem, output->im.reg, x, y);
    uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->im.mem, param1->im.reg, x, y);

    output->im.mem.ptr[index] = param1->im.mem.ptr[idx1] + value;
  }
}

__hostdeviceinline__ void cuda_GenericApi_addRealConst (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[3];

  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    math::ComplexMatrix* output = outputs[oidxs[0]];
    const math::ComplexMatrix* param1 = params1[oidxs[0]];

    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    {
      uintt index = oap::common::GetMemIdxFromMatrixPos (output->re.mem, output->re.reg, x, y);
      uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->re.mem, param1->re.reg, x, y);
      output->re.mem.ptr[index] = param1->re.mem.ptr[idx1] + value;
    }
    {
      uintt index = oap::common::GetMemIdxFromMatrixPos (output->im.mem, output->im.reg, x, y);
      uintt idx1 = oap::common::GetMemIdxFromMatrixPos (param1->im.mem, param1->im.reg, x, y);
      output->im.mem.ptr[index] = param1->im.mem.ptr[idx1] + value;
    }
  }
}

__hostdeviceinline__ void CUDA_GenericApi_addReMatrixConst (math::ComplexMatrix** output, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addReConst (output, params1, value, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addImMatrixConst (math::ComplexMatrix** output, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addImConst (output, params1, value, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addRealMatrixConst (math::ComplexMatrix** output, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addRealConst (output, params1, value, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_AddConst (math::ComplexMatrix** output, math::ComplexMatrix* const* params1, floatt value, oap::ThreadsMapperS* mapper)
{
  bool isRe = output[0]->re.mem.ptr != NULL;
  bool isIm = output[0]->im.mem.ptr != NULL;

  if (isRe && isIm)
  {
    CUDA_GenericApi_addRealMatrixConst (output, params1, value, mapper);
  }
  else if (isRe)
  {
    CUDA_GenericApi_addReMatrixConst (output, params1, value, mapper);
  }
  else if (isIm)
  {
    CUDA_GenericApi_addImMatrixConst (output, params1, value, mapper);
  }
}

#endif
