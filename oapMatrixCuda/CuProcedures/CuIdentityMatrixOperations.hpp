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

#ifndef OAP_CU_IDENTITY_MATRIX_OPERATIONS_H
#define OAP_CU_IDENTITY_MATRIX_OPERATIONS_H

#include "CuCore.hpp"
#include "MatrixAPI.hpp"

__hostdevice__ void cuda_IdentityMatrixAdd_Real (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const floatt re = GetRe (matrix, threadIndexX, threadIndexY);
  const floatt im = GetIm (matrix, threadIndexX, threadIndexY);

  SetRe (output, threadIndexX, threadIndexY, re);
  SetIm (output, threadIndexX, threadIndexY, im);

  const bool isDiag = threadIndexX == threadIndexY;

  if (isDiag)
  {
    const floatt re1 = 1. + GetRe (output, threadIndexX, threadIndexY);
    const floatt im1 = 1. + GetIm (output, threadIndexX, threadIndexY);

    SetRe (output, threadIndexX, threadIndexY, re1);
    SetIm (output, threadIndexX, threadIndexY, im1);
  }
}

__hostdevice__ void cuda_IdentityMatrixAdd_Re (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const floatt re = GetRe (matrix, threadIndexX, threadIndexY);

  SetRe (output, threadIndexX, threadIndexY, re);

  const bool isDiag = threadIndexX == threadIndexY;

  if (isDiag)
  {
    const floatt re1 = 1. + GetRe (output, threadIndexX, threadIndexY);

    SetRe (output, threadIndexX, threadIndexY, re1);
  }
}

__hostdevice__ void CUDA_IdentityMatrixAdd (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const bool inScope = threadIndexX < gColumns (output) && threadIndexY < gRows (output); 

  if (inScope)
  {
    const bool isRe = output->re.mem.ptr != NULL;
    const bool isIm = output->im.mem.ptr != NULL;
  
    if (isRe && isIm)
    {
      cuda_IdentityMatrixAdd_Real (output, matrix);
    }
    else if (isRe)
    {
      cuda_IdentityMatrixAdd_Re (output, matrix);
    }
    else if (isIm)
    {
      assert ("not supported operation" == NULL);
    }
  }
  threads_sync ();
}

__hostdevice__ void cuda_IdentityMatrixSubstract_Real (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const floatt re = GetRe (matrix, threadIndexX, threadIndexY);
  const floatt im = GetIm (matrix, threadIndexX, threadIndexY);

  SetRe (output, threadIndexX, threadIndexY, -re);
  SetIm (output, threadIndexX, threadIndexY, -im);

  const bool isDiag = threadIndexX == threadIndexY;

  if (isDiag)
  {
    const floatt re1 = 1. + GetRe (output, threadIndexX, threadIndexY);
    const floatt im1 = 1. + GetIm (output, threadIndexX, threadIndexY);

    SetRe (output, threadIndexX, threadIndexY, re1);
    SetIm (output, threadIndexX, threadIndexY, im1);
  }
}

__hostdevice__ void cuda_IdentityMatrixSubstract_Re (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const floatt re = GetRe (matrix, threadIndexX, threadIndexY);

  SetRe (output, threadIndexX, threadIndexY, -re);

  const bool isDiag = threadIndexX == threadIndexY;

  if (isDiag)
  {
    const floatt re1 = 1. + GetRe (output, threadIndexX, threadIndexY);

    SetRe (output, threadIndexX, threadIndexY, re1);
  }
}

__hostdevice__ void CUDA_IdentityMatrixSubstract (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const bool inScope = threadIndexX < gColumns (output) && threadIndexY < gRows (output); 

  if (inScope)
  {
    const bool isRe = output->re.mem.ptr != NULL;
    const bool isIm = output->im.mem.ptr != NULL;
  
    if (isRe && isIm)
    {
      cuda_IdentityMatrixSubstract_Real (output, matrix);
    }
    else if (isRe)
    {
      cuda_IdentityMatrixSubstract_Re (output, matrix);
    }
    else if (isIm)
    {
      assert ("not supported operation" == NULL);
    }
  }
  threads_sync ();
}

#endif
