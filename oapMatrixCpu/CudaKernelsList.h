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

#ifndef OAP_CUDA_KERNELS_LIST_H
#define OAP_CUDA_KERNELS_LIST_H

#include "CuMatrixProcedures.h"

template<typename T>
T* getParam (void* param)
{
  return *static_cast<T**> (param);
}

template<typename T>
T* getParam (void** params, size_t index)
{
  return getParam<T> (params[index]);
}

void HOSTKernel_SumShared (floatt* rebuffer, floatt* imbuffer, math::Matrix* matrix)
{
  CUDA_sumShared (rebuffer, imbuffer, matrix);
}

void HOSTKernel_SumSharedRaw (void** params)
{
  floatt* param1 = getParam<floatt> (params[0]);
  floatt* param2 = getParam<floatt> (params[1]);
  math::Matrix* param3 = getParam<math::Matrix> (params[2]);
  HOSTKernel_SumShared (param1, param2, param3);
}

void HOSTKernel_CrossEntropy (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  CUDA_crossEntropy (output, param1, param2);
}

void HOSTKernel_Tanh (math::Matrix* output, math::Matrix* param1)
{
  CUDA_tanh (output, param1);
}

void HOSTKernel_TanhRaw (void** params)
{
  math::Matrix* output = getParam<math::Matrix> (params[0]);
  math::Matrix* matrix = getParam<math::Matrix> (params[1]);
  HOSTKernel_Tanh (output, matrix);
}

void HOSTKernel_Sigmoid (math::Matrix* output, math::Matrix* param1)
{
  CUDA_sigmoid (output, param1);
}

void HOSTKernel_SigmoidRaw (void** params)
{
  math::Matrix* output = getParam<math::Matrix> (params[0]);
  math::Matrix* matrix = getParam<math::Matrix> (params[1]);
  HOSTKernel_Sigmoid (output, matrix);
}

void HOSTKernel_CrossEntropyRaw (void** params)
{
  math::Matrix* output = getParam<math::Matrix> (params[0]);
  math::Matrix* param1 = getParam<math::Matrix> (params[1]);
  math::Matrix* param2 = getParam<math::Matrix> (params[2]);

  HOSTKernel_CrossEntropy (output, param1, param2);
}

void HOSTKernel_DotProduct (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  CUDA_dotProduct (output, param1, param2);
}

void HOSTKernel_DotProductRaw (void** params)
{
  math::Matrix* output = getParam<math::Matrix> (params[0]);
  math::Matrix* matrix1 = getParam<math::Matrix> (params[1]);
  math::Matrix* matrix2 = getParam<math::Matrix> (params[2]);
  HOSTKernel_DotProduct (output, matrix1, matrix2);
}

void HOSTKernel_DotProductDim (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, uintt* ex)
{
  CUDA_dotProductDim (output, param1, param2, ex);
}

void HOSTKernel_DotProductDimRaw (void** params)
{
  math::Matrix* output = getParam<math::Matrix> (params[0]);
  math::Matrix* matrix1 = getParam<math::Matrix> (params[1]);
  math::Matrix* matrix2 = getParam<math::Matrix> (params[2]);
  uintt* ex = getParam<uintt> (params[3]);
  HOSTKernel_DotProductDim (output, matrix1, matrix2, ex);
}

void HOSTKernel_TensorProductDim (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, uintt* ex)
{
  CUDA_tensorProductDim (output, param1, param2, ex);
}

void HOSTKernel_TensorProductDimRaw (void** params)
{
  math::Matrix* output = getParam<math::Matrix> (params[0]);
  math::Matrix* matrix1 = getParam<math::Matrix> (params[1]);
  math::Matrix* matrix2 = getParam<math::Matrix> (params[2]);
  uintt* ex = getParam<uintt> (params[3]);
  HOSTKernel_TensorProductDim (output, matrix1, matrix2, ex);
}

#endif
