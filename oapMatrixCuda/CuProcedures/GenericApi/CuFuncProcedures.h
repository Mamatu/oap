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

#ifndef OAP_API2_CU_FUNC_PROCEDURES_H
#define OAP_API2_CU_FUNC_PROCEDURES_H

#include "../CuFuncTypes.h"
#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"
#include "oapThreadsMapperS.h"
#include "oapMemory_ThreadMapperApi.h"

__hostdeviceinline__ void cuda_funcRe (math::Matrix** outputs, math::Matrix* const* params, func_t func, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param = params[oidxs[0]];

    uintt idx = oap::common::GetMemIdxFromMatrixPos (output->re, output->reReg, x, y);

    floatt* outputValue = &output->re.ptr[idx];
    func (outputValue, param->re.ptr[idx]);
  }
}

__hostdeviceinline__ void cuda_funcIm (math::Matrix** outputs, math::Matrix* const* params, func_t func, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param = params[oidxs[0]];

    uintt idx = oap::common::GetMemIdxFromMatrixPos (output->im, output->imReg, x, y);

    floatt* outputValue = &output->im.ptr[idx];
    func (outputValue, param->im.ptr[idx]);
  }
}

__hostdeviceinline__ void cuda_funcReal (math::Matrix** outputs, math::Matrix* const* params, func_t func, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param = params[oidxs[0]];

    {
      uintt idx = oap::common::GetMemIdxFromMatrixPos (output->re, output->reReg, x, y);
      floatt* outputValue = &output->re.ptr[idx];
      func (outputValue, param->re.ptr[idx]);
    }
    {
      uintt idx = oap::common::GetMemIdxFromMatrixPos (output->im, output->imReg, x, y);
      floatt* outputValue = &output->im.ptr[idx];
      func (outputValue, param->im.ptr[idx]);
    }
  }
}

__hostdeviceinline__ void cuda_funcRe_userData (math::Matrix** outputs, math::Matrix* const* params, func_ud_t func, void* ud, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param = params[oidxs[0]];

    uintt idx = oap::common::GetMemIdxFromMatrixPos (output->re, output->reReg, x, y);

    floatt* outputValue = &output->re.ptr[idx];
    func (outputValue, param->re.ptr[idx], ud);
  }
}

__hostdeviceinline__ void cuda_funcIm_userData (math::Matrix** outputs, math::Matrix* const* params, func_ud_t func, void* ud, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param = params[oidxs[0]];

    uintt idx = oap::common::GetMemIdxFromMatrixPos (output->im, output->imReg, x, y);

    floatt* outputValue = &output->im.ptr[idx];
    func (outputValue, param->im.ptr[idx], ud);
  }
}

__hostdeviceinline__ void cuda_funcReal_userData (math::Matrix** outputs, math::Matrix* const* params, func_ud_t func, void* ud, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
 
  uintt oidxs[3];
  bool inrange = _idxpos_check(oidxs, outputs, mapper, 0);

  if (inrange)
  {
    const uintt x = oidxs[1];
    const uintt y = oidxs[2];

    math::Matrix* output = outputs[oidxs[0]];
    math::Matrix* param = params[oidxs[0]];

    {
      uintt idx = oap::common::GetMemIdxFromMatrixPos (output->re, output->reReg, x, y);
      floatt* outputValue = &output->re.ptr[idx];
      func (outputValue, param->re.ptr[idx], ud);
    }
    {
      uintt idx = oap::common::GetMemIdxFromMatrixPos (output->im, output->imReg, x, y);
      floatt* outputValue = &output->im.ptr[idx];
      func (outputValue, param->im.ptr[idx], ud);
    }
  }
}


__hostdeviceinline__ void cuda_func (math::Matrix** outputs, math::Matrix* const* params, func_t func, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  bool isre = outputs[0]->re.ptr != NULL;
  bool isim = outputs[0]->im.ptr != NULL;

  if (isre && isim)
  {
    cuda_funcReal (outputs, params, func, mapper);
  }
  else if (isre)
  {
    cuda_funcRe (outputs, params, func, mapper);
  }
  else if (isim)
  {
    cuda_funcIm (outputs, params, func, mapper);
  }
}

__hostdeviceinline__ void cuda_func_userData (math::Matrix** outputs, math::Matrix* const* params, func_ud_t func, void* ud, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  bool isre = outputs[0]->re.ptr != NULL;
  bool isim = outputs[0]->im.ptr != NULL;

  if (isre && isim)
  {
    cuda_funcReal_userData (outputs, params, func, ud, mapper);
  }
  else if (isre)
  {
    cuda_funcRe_userData (outputs, params, func, ud, mapper);
  }
  else if (isim)
  {
    cuda_funcIm_userData (outputs, params, func, ud, mapper);
  }
}

#endif
