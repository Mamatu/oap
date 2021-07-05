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

#ifndef CU_FUNC_PROCEDURES_H
#define CU_FUNC_PROCEDURES_H

#include "CuCore.hpp"
#include "CuUtils.hpp"
#include "Matrix.hpp"

typedef void(*func_t)(floatt*, floatt);
typedef void(*func_ud_t)(floatt*, floatt, void*);

__hostdeviceinline__ void cuda_funcRe (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_t func)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (omatrix);
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* output = &*GetRePtrIndex (omatrix, index);
  func (output, *GetRePtrIndex (imatrix, index));
}

__hostdeviceinline__ void cuda_funcIm (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_t func)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (omatrix);
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* output = &*GetImPtrIndex (omatrix, index);
  func (output, *GetImPtrIndex (imatrix, index));
}

__hostdeviceinline__ void cuda_funcReal (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_t func)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (omatrix);
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* reoutput = &*GetRePtrIndex (omatrix, index);
  floatt* imoutput = &*GetImPtrIndex (omatrix, index);
  func (reoutput, *GetRePtrIndex (imatrix, index));
  func (imoutput, *GetImPtrIndex (imatrix, index));
}

__hostdeviceinline__ void cuda_funcRe_userData (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_ud_t func, void* ud)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (omatrix);
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* output = &*GetRePtrIndex (omatrix, index);
  func (output, *GetRePtrIndex (imatrix, index), ud);
  //cuda_debug ("output = %f ");
}

__hostdeviceinline__ void cuda_funcIm_userData (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_ud_t func, void* ud)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (omatrix);
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* output = &*GetImPtrIndex (omatrix, index);
  func (output, *GetImPtrIndex (imatrix, index), ud);
}

__hostdeviceinline__ void cuda_funcReal_userData (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_ud_t func, void* ud)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (omatrix);
  uintt index = threadIndexX + offset * threadIndexY;

  floatt* reoutput = &*GetRePtrIndex (omatrix, index);
  floatt* imoutput = &*GetImPtrIndex (omatrix, index);
  func (reoutput, *GetRePtrIndex (imatrix, index), ud);
  func (imoutput, *GetImPtrIndex (imatrix, index), ud);
}

__hostdeviceinline__ void cuda_func (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_t func)
{
  HOST_INIT();

  bool isre = omatrix->re.mem.ptr != NULL;
  bool isim = omatrix->im.mem.ptr != NULL;

  if (isre && isim)
  {
    cuda_funcReal (omatrix, imatrix, func);
  }
  else if (isre)
  {
    cuda_funcRe (omatrix, imatrix, func);
  }
  else if (isim)
  {
    cuda_funcIm (omatrix, imatrix, func);
  }
}

__hostdeviceinline__ void cuda_func_userData (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, func_ud_t func, void* ud)
{
  HOST_INIT();

  bool isre = omatrix->re.mem.ptr != NULL;
  bool isim = omatrix->im.mem.ptr != NULL;

  if (isre && isim)
  {
    cuda_funcReal_userData (omatrix, imatrix, func, ud);
  }
  else if (isre)
  {
    cuda_funcRe_userData (omatrix, imatrix, func, ud);
  }
  else if (isim)
  {
    cuda_funcIm_userData (omatrix, imatrix, func, ud);
  }
}

__hostdeviceinline__ bool cuda_inRangePD (math::ComplexMatrix* matrix, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt indexY1 = (threadIndexY) % ex[2];
  return threadIndexX < ex[0] && threadIndexY < gRows (matrix) && indexY1 < ex[1];
}

#endif
