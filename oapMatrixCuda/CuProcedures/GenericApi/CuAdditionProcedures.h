/*
 * Copyright 2016 - 2019 Marcin Matula
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

#define GET_REIDX(matrix) (threadIndexX + matrix->reReg.loc.x) + matrix->re.dims.width * (threadIndexY + matrix->reReg.loc.y)

#define GET_IMIDX(matrix) (threadIndexX + matrix->imReg.loc.x) + matrix->im.dims.width * (threadIndexY + matrix->imReg.loc.y)

#define GET_IDX(region) (threadIndexX + region.loc.x) + stride * (threadIndexY + region.loc.y)

#define GET_LEN(reg) reg.dims.width * reg.dims.height

__hostdeviceinline__ void cuda_GenericApi_addReMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, oap::ThreadMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt stride = GetColumns (output[0]);
  uintt threadIndex = threadIndexY * stride + threadIndexX;

  threadIndex = mapper[threadIndex];
  if (threadIndex < MAX_UINTT)
  {
    math::Matrix* o = output[threadIndex];
    const math::Matrix* p0 = params0[threadIndex];

    uintt oidx = oap::common::GetIdx (o->re, o->reReg, threadIndexX, threadIndexY);
    uintt p0idx = oap::common::GetIdx (p0->re, p0->reReg, threadIndexX, threadIndexY);
    //uintt oidx = GET_REIDX(o);
    //uintt p0idx = GET_REIDX(p0);

    HOST_CODE(oapAssert(oidx < GET_LEN(o->re)));
    HOST_CODE(oapAssert(p0idx < GET_LEN(p0->re)));

    o->re.ptr[oidx] = p0->re.ptr[p0idx] + params1;
  }
}

__hostdeviceinline__ void cuda_GenericApi_addImMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, uintt* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt stride = GetColumns (output[0]);
  uintt threadIndex = threadIndexY * stride + threadIndexX;

  threadIndex = mapper[threadIndex];
  if (threadIndex < MAX_UINTT)
  {
    math::Matrix* o = output[threadIndex];
    const math::Matrix* p0 = params0[threadIndex];

    uintt oidx = GET_IMIDX(o);
    uintt p0idx = GET_IMIDX(p0);

    HOST_CODE(oapAssert(oidx < GET_LEN(o->im)));
    HOST_CODE(oapAssert(p0idx < GET_LEN(p0->im)));

    o->im.ptr[oidx] = p0->im.ptr[p0idx] + params1;
  }
}

__hostdeviceinline__ void cuda_GenericApi_addRealMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, uintt* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt stride = GetColumns (output[0]);
  uintt threadIndex = threadIndexY * stride + threadIndexX;

  threadIndex = mapper[threadIndex];
  if (threadIndex < MAX_UINTT)
  {
    math::Matrix* o = output[threadIndex];
    const math::Matrix* p0 = params0[threadIndex];

    uintt reoidx = GET_REIDX(o);
    uintt rep0idx = GET_REIDX(p0);

    uintt imoidx = GET_IMIDX(o);
    uintt imp0idx = GET_IMIDX(p0);

    HOST_CODE(oapAssert(reoidx < GET_LEN(o->re)));
    HOST_CODE(oapAssert(rep0idx < GET_LEN(p0->re)));
    HOST_CODE(oapAssert(imoidx < GET_LEN(o->im)));
    HOST_CODE(oapAssert(imp0idx < GET_LEN(p0->im)));

    o->re.ptr[reoidx] = p0->re.ptr[rep0idx] + params1;
    o->im.ptr[imoidx] = p0->im.ptr[imp0idx] + params1;
  }
}

__hostdeviceinline__ void CUDA_GenericApi_addReMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, uintt* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addReMatrixValue (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addImMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, uintt* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addImMatrixValue (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addRealMatrixValue (math::Matrix** output, math::Matrix* const* params0, floatt params1, uintt* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addRealMatrixValue (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_AddConstant (math::Matrix** output, math::Matrix* const* params0, floatt params1, uintt* mapper)
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
