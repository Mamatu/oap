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

#define DEFINE_1M(arg_hostKernel, arg_cudaKernel)                 \
void arg_hostKernel (math::Matrix* output)                        \
{                                                                 \
  arg_cudaKernel (output);                                        \
}                                                                 \
                                                                  \
void proxy_##arg_hostKernel (void** params)                       \
{                                                                 \
  math::Matrix* output = getParam<math::Matrix> (params[0]);      \
  arg_hostKernel (output);                                        \
}

#define DEFINE_1M_EX(arg_hostKernel, arg_cudaKernel)                          \
void arg_hostKernel (math::Matrix* output, uintt* ex)                         \
{                                                                             \
  arg_cudaKernel (output, ex);                                                \
}                                                                             \
                                                                              \
void proxy_##arg_hostKernel (void** params)                                   \
{                                                                             \
  math::Matrix* output = getParam<math::Matrix> (params[0]);                  \
  uintt* ex = getParam<uintt> (params[1]);                                    \
  arg_hostKernel (output, ex);                                                \
}

#define DEFINE_2M(arg_hostKernel, arg_cudaKernel)                 \
void arg_hostKernel (math::Matrix* output, math::Matrix* param1)  \
{                                                                 \
  arg_cudaKernel (output, param1);                                \
}                                                                 \
                                                                  \
void proxy_##arg_hostKernel (void** params)                       \
{                                                                 \
  math::Matrix* output = getParam<math::Matrix> (params[0]);      \
  math::Matrix* matrix = getParam<math::Matrix> (params[1]);      \
  arg_hostKernel (output, matrix);                                \
}

#define DEFINE_2M_EX(arg_hostKernel, arg_cudaKernel)                          \
void arg_hostKernel (math::Matrix* output, math::Matrix* param1, uintt* ex)   \
{                                                                             \
  arg_cudaKernel (output, param1, ex);                                        \
}                                                                             \
                                                                              \
void proxy_##arg_hostKernel (void** params)                                   \
{                                                                             \
  math::Matrix* output = getParam<math::Matrix> (params[0]);                  \
  math::Matrix* matrix = getParam<math::Matrix> (params[1]);                  \
  uintt* ex = getParam<uintt> (params[2]);                                    \
  arg_hostKernel (output, matrix, ex);                                        \
}

#define DEFINE_3M(arg_hostKernel, arg_cudaKernel)                                       \
void arg_hostKernel (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)  \
{                                                                                       \
  arg_cudaKernel (output, param1, param2);                                              \
}                                                                                       \
                                                                                        \
void proxy_##arg_hostKernel (void** params)                                             \
{                                                                                       \
  math::Matrix* output = getParam<math::Matrix> (params[0]);                            \
  math::Matrix* matrix = getParam<math::Matrix> (params[1]);                            \
  math::Matrix* matrix1 = getParam<math::Matrix> (params[2]);                           \
  arg_hostKernel (output, matrix, matrix1);                                             \
}

#define DEFINE_3M_EX(arg_hostKernel, arg_cudaKernel)                                                \
void arg_hostKernel (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, uintt* ex)   \
{                                                                                                   \
  arg_cudaKernel (output, param1, param2, ex);                                                      \
}                                                                                                   \
                                                                                                    \
void proxy_##arg_hostKernel (void** params)                                                         \
{                                                                                                   \
  math::Matrix* output = getParam<math::Matrix> (params[0]);                                        \
  math::Matrix* matrix = getParam<math::Matrix> (params[1]);                                        \
  math::Matrix* matrix1 = getParam<math::Matrix> (params[2]);                                       \
  uintt* ex = getParam<uintt> (params[3]);                                                          \
  arg_hostKernel (output, matrix, matrix1, ex);                                                     \
}

void HOSTKernel_SumShared (floatt* rebuffer, floatt* imbuffer, math::Matrix* matrix)
{
  CUDA_sumShared (rebuffer, imbuffer, matrix);
}

void proxy_HOSTKernel_SumShared (void** params)
{
  floatt* param1 = getParam<floatt> (params[0]);
  floatt* param2 = getParam<floatt> (params[1]);
  math::Matrix* param3 = getParam<math::Matrix> (params[2]);
  HOSTKernel_SumShared (param1, param2, param3);
}

DEFINE_1M(HOSTKernel_SetIdentity, CUDA_setIdentityMatrix)

DEFINE_3M(HOSTKernel_CrossEntropy, CUDA_crossEntropy)
DEFINE_3M(HOSTKernel_DotProduct, CUDA_dotProduct)
//DEFINE_3M(HOSTKernel_DotProductShared, CUDAKernel_dotProductShared)

DEFINE_2M(HOSTKernel_Tanh, CUDA_tanh)
DEFINE_2M(HOSTKernel_Sigmoid, CUDA_sigmoid)
DEFINE_2M(HOSTKernel_Sin, CUDA_sin)

DEFINE_2M_EX(HOSTKernel_TanhDim, CUDA_tanhDim)
DEFINE_2M_EX(HOSTKernel_SigmoidDim, CUDA_sigmoidDim)
DEFINE_2M_EX(HOSTKernel_SinDim, CUDA_sinDim)

DEFINE_2M_EX(HOSTKernel_TanhDimPeriodic, CUDA_tanhDimPeriodic)
DEFINE_2M_EX(HOSTKernel_SigmoidDimPeriodic, CUDA_sigmoidDimPeriodic)
DEFINE_2M_EX(HOSTKernel_SinDimPeriodic, CUDA_sinDimPeriodic)

DEFINE_3M_EX(HOSTKernel_DotProductDim, CUDA_dotProductDim)
DEFINE_3M(HOSTKernel_DotProductPeriodic, CUDA_dotProductPeriodic)
DEFINE_3M_EX(HOSTKernel_DotProductDimPeriodic, CUDA_dotProductDimPeriodic)

DEFINE_3M_EX(HOSTKernel_TensorProductDim, CUDA_tensorProductDim)

void HOSTKernel_QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT)
{
  CudaKernel_QRHT(Q, R, A, V, VT, P, VVT);
}

void proxy_HOSTKernel_QRHT (void** params)
{
  size_t i = 0;
  math::Matrix* Q = getParam<math::Matrix> (params[i++]);
  math::Matrix* R = getParam<math::Matrix> (params[i++]);
  math::Matrix* A = getParam<math::Matrix> (params[i++]);
  math::Matrix* V = getParam<math::Matrix> (params[i++]);
  math::Matrix* VT = getParam<math::Matrix> (params[i++]);
  math::Matrix* P = getParam<math::Matrix> (params[i++]);
  math::Matrix* VVT = getParam<math::Matrix> (params[i++]);

  HOSTKernel_QRHT (Q, R, A, V, VT, P, VVT);
}

#endif
