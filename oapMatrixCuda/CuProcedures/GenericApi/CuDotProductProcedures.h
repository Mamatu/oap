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

#ifndef OAP_API2_CU_DOT_PRODUCT_PROCEDURES_H
#define OAP_API2_CU_DOT_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

__hostdevice__ void cuda_GenericApi_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = gColumns (params0);
  const uintt columns2 = gColumns (params1);
  const uintt offset = columns1;

  floatt retemp = 0;

  for (intt midx = 0; midx < offset; midx++)
  {
    retemp += gReValues (params0)[midx + columns1 * threadIndexY] * gReValues (params1)[midx * columns2 + threadIndexX];
  }

  *GetRePtrIndex (output, threadIndexX + gColumns (output) * threadIndexY) = retemp;
}

__hostdevice__ void cuda_GenericApi_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = gColumns (params0);
  const uintt columns2 = gColumns (params1);
  const uintt offset = columns1;

  floatt retemp = 0;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    retemp += -gImValues (params0)[midx + columns1 * threadIndexY] * gImValues (params1)[midx * columns2 + threadIndexX];
  }

  *GetRePtrIndex (output, threadIndexX + gColumns (output) * threadIndexY) = retemp;
}

__hostdevice__ void cuda_GenericApi_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = gColumns (params0);
  const uintt columns2 = gColumns (params1);
  const uintt outputColumns = gColumns (output);
  const uintt offset = columns1;

  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt midx = 0; midx < offset; midx++)
  {
    retemp += gReValues (params0)[midx + columns1 * threadIndexY] *
              gReValues (params1)[midx * columns2 + threadIndexX];
    retemp -= gImValues (params0)[midx + columns1 * threadIndexY] *
              gImValues (params1)[midx * columns2 + threadIndexX];
    imtemp += gReValues (params0)[midx + columns1 * threadIndexY] *
              gImValues (params1)[midx * columns2 + threadIndexX];
    imtemp += gImValues (params0)[midx + columns1 * threadIndexY] *
              gReValues (params1)[midx * columns2 + threadIndexX];
  }

  *GetRePtrIndex (output, threadIndexX + outputColumns * threadIndexY) = retemp;
  gImValues (output)[threadIndexX + outputColumns * threadIndexY] = imtemp;
}

__hostdevice__ void CUDA_GenericApi_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  cuda_GenericApi_dotProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  cuda_GenericApi_dotProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  cuda_GenericApi_dotProductReal(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProduct (oap::Memory& output, const oap::Memory& arg1, const oap::Memory& arg2, const oap::MemoryRegion_3_Args* regions, uintt regionsCount)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

}

#endif
