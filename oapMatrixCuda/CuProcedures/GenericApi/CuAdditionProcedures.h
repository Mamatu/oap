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

#ifndef OAP_API2_CU_ADDITION_PROCEDURES_H
#define OAP_API2_CU_ADDITION_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "oapMemory_ThreadMapperApi.h"
#include "oapThreadsMapperS.h"

#define _getReIdx(matrix) (threadIndexX + matrix->reReg.loc.x) + matrix->re.dims.width * (threadIndexY + matrix->reReg.loc.y)

#define _getImIdx(matrix) (threadIndexX + matrix->imReg.loc.x) + matrix->im.dims.width * (threadIndexY + matrix->imReg.loc.y)

#define _getIdx(region) (threadIndexX + region.loc.x) + stride * (threadIndexY + region.loc.y)

#define _getLen(reg) reg.dims.width * reg.dims.height

__hostdeviceinline__ void cuda_GenericApi_addReMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[2];
  uintt pidxs[2];

  _idxs(oidxs, output, mapper, 0);
  _idxs(pidxs, params0, mapper, 1);

  if (oidxs[0] < MAX_UINTT)
  {
    math::Matrix* o = output[oidxs[0]];
    const math::Matrix* p0 = params0[pidxs[0]];

    HOST_CODE(oapAssert(oidxs[1] < _getLen(o->re)));
    HOST_CODE(oapAssert(pidxs[1] < _getLen(p0->re)));

    o->re.ptr[oidxs[1]] = p0->re.ptr[pidxs[1]] + params1;
  }
}

__hostdeviceinline__ void cuda_GenericApi_addImMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[2];
  uintt pidxs[2];

  _idxs(oidxs, output, mapper, 0);
  _idxs(pidxs, params0, mapper, 1);

  if (oidxs[0] < MAX_UINTT)
  {
    math::Matrix* o = output[oidxs[0]];
    const math::Matrix* p0 = params0[pidxs[0]];

    HOST_CODE(oapAssert(oidxs[1] < _getLen(o->re)));
    HOST_CODE(oapAssert(pidxs[1] < _getLen(p0->re)));

    o->im.ptr[oidxs[1]] = p0->im.ptr[pidxs[1]] + params1;
  }
}

__hostdeviceinline__ void cuda_GenericApi_addRealMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[2];
  uintt pidxs[2];

  _idxs(oidxs, output, mapper, 0);
  _idxs(pidxs, params0, mapper, 1);

  if (oidxs[0] < MAX_UINTT)
  {
    math::Matrix* o = output[oidxs[0]];
    const math::Matrix* p0 = params0[pidxs[0]];

    HOST_CODE(oapAssert(oidxs[1] < _getLen(o->re)));
    HOST_CODE(oapAssert(pidxs[1] < _getLen(p0->re)));

    o->re.ptr[oidxs[1]] = p0->re.ptr[pidxs[1]] + params1;
    o->im.ptr[oidxs[1]] = p0->im.ptr[pidxs[1]] + params1;
  }
}

__hostdeviceinline__ void CUDA_GenericApi_addReMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addReMatrixValue (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addImMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addImMatrixValue (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addRealMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addRealMatrixValue (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_AddConstant (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadsMapperS* mapper)
{
  bool isRe = output[0]->re.ptr != NULL;
  bool isIm = output[0]->im.ptr != NULL;

  if (isRe && isIm)
  {
    CUDA_GenericApi_addRealMatrixValue (output, params0, params1, mapper);
  }
  else if (isRe)
  {
    CUDA_GenericApi_addReMatrixValue (output, params0, params1, mapper);
  }
  else if (isIm)
  {
    CUDA_GenericApi_addImMatrixValue (output, params0, params1, mapper);
  }
}

#endif
