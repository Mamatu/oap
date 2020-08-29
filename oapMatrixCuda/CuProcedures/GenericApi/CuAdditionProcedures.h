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
#include "oapThreadsMapperS.h"

__hostdeviceinline__ void cuda_GenericApi_addReMatrix (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[2];
  uintt p0idxs[2];
  uintt p1idxs[2];

  _idxs(oidxs, output, mapper, 0);
  _idxs(p0idxs, params0, mapper, 1);
  _idxs(p1idxs, params1, mapper, 2);

  if (oidxs[0] < MAX_UINTT)
  {
    math::Matrix* o = output[oidxs[0]];
    const math::Matrix* p0 = params0[p0idxs[0]];
    const math::Matrix* p1 = params1[p1idxs[0]];

    HOST_CODE(oapAssert(oidxs[1] < _getLen(o->re)));
    HOST_CODE(oapAssert(p0idxs[1] < _getLen(p0->re)));
    HOST_CODE(oapAssert(p1idxs[1] < _getLen(p1->re)));

    o->re.ptr[oidxs[1]] = p0->re.ptr[p0idxs[1]] + p1->re.ptr[p1idxs[1]];
  }
}

__hostdeviceinline__ void cuda_GenericApi_addImMatrix (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[2];
  uintt p0idxs[2];
  uintt p1idxs[2];

  _idxs(oidxs, output, mapper, 0);
  _idxs(p0idxs, params0, mapper, 1);
  _idxs(p1idxs, params1, mapper, 2);

  if (oidxs[0] < MAX_UINTT)
  {
    math::Matrix* o = output[oidxs[0]];
    const math::Matrix* p0 = params0[p0idxs[0]];
    const math::Matrix* p1 = params1[p1idxs[0]];

    HOST_CODE(oapAssert(oidxs[1] < _getLen(o->re)));
    HOST_CODE(oapAssert(p0idxs[1] < _getLen(p0->re)));
    HOST_CODE(oapAssert(p1idxs[1] < _getLen(p1->re)));

    o->im.ptr[oidxs[1]] = p0->im.ptr[p0idxs[1]] + p1->im.ptr[p1idxs[1]];
  }
}

__hostdeviceinline__ void cuda_GenericApi_addRealMatrix (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[2];
  uintt p0idxs[2];
  uintt p1idxs[2];

  _idxs(oidxs, output, mapper, 0);
  _idxs(p0idxs, params0, mapper, 1);
  _idxs(p1idxs, params1, mapper, 2);

  if (oidxs[0] < MAX_UINTT)
  {
    math::Matrix* o = output[oidxs[0]];
    const math::Matrix* p0 = params0[p0idxs[0]];
    const math::Matrix* p1 = params1[p1idxs[0]];

    HOST_CODE(oapAssert(oidxs[1] < _getLen(o->re)));
    HOST_CODE(oapAssert(p0idxs[1] < _getLen(p0->re)));
    HOST_CODE(oapAssert(p1idxs[1] < _getLen(p1->re)));

    o->re.ptr[oidxs[1]] = p0->re.ptr[p0idxs[1]] + p1->re.ptr[p1idxs[1]];
    o->im.ptr[oidxs[1]] = p0->im.ptr[p0idxs[1]] + p1->im.ptr[p1idxs[1]];
  }
}

__hostdeviceinline__ void CUDA_GenericApi_addReMatrix (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addReMatrix (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addImMatrix (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addImMatrix (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_addRealMatrix (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  cuda_GenericApi_addRealMatrix (output, params0, params1, mapper);
  threads_sync();
}

__hostdeviceinline__ void CUDA_GenericApi_Add (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, oap::ThreadsMapperS* mapper)
{
  bool isRe = output[0]->re.ptr != NULL;
  bool isIm = output[0]->im.ptr != NULL;

  if (isRe && isIm)
  {
    CUDA_GenericApi_addRealMatrix (output, params0, params1, mapper);
  }
  else if (isRe)
  {
    CUDA_GenericApi_addReMatrix (output, params0, params1, mapper);
  }
  else if (isIm)
  {
    CUDA_GenericApi_addImMatrix (output, params0, params1, mapper);
  }
}

#endif
