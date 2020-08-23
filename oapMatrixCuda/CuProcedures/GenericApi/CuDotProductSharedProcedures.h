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

#ifndef OAP_API2_CU_DOT_PRODUCT_SHARED_PROCEDURES_H
#define OAP_API2_CU_DOT_PRODUCT_SHARED_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "oapMemory_ThreadMapperApi.h"
#include "oapThreadsMapperS.h"

#include "../CuCreateProcedures.h"
#include "../CuDotProductUtils.h"
#if 0
__hostdevice__ void cuda_GenericApi_dotProductRe (math::Matrix** outputs, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt oidxs[3];
  _idxpos(oidxs, outputs, mapper, 0);

  math::Matrix* output = outputs[oidxs[0]];
  math::Matrix* param0 = params0[oidxs[0]];
  math::Matrix* param1 = params1[oidxs[0]];

  MatrixEx cexs[3];
  cuAux_initMatrixExs (cexs, output, param0, param1);

  MatrixOffset matrixOffset0 = CUDA_createMatrixCopy (sbuffer, param0, cexs[1]);
  MatrixOffset matrixOffset1 = CUDA_createMatrixCopy (matrixOffset0.buffer, param1, cexs[2]);

  const math::Matrix& sharedParams1 = matrixOffset1.matrix;
}

__hostdevice__ void cuda_GenericApi_dotProductIm  (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void cuda_GenericApi_dotProductReal (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void CUDA_GenericApi_dotProductRe (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  cuda_GenericApi_dotProductRe(output, params0, params1, sbuffer, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProductIm (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  cuda_GenericApi_dotProductIm(output, params0, params1, sbuffer, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProductReal (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();

  cuda_GenericApi_dotProductReal(output, params0, params1, sbuffer, mapper);
  threads_sync();
}

__hostdevice__ void CUDA_GenericApi_dotProduct (math::Matrix** output, math::Matrix* const* params0, math::Matrix* const* params1, floatt* sbuffer, oap::ThreadsMapperS* mapper)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

}
#endif
#endif
