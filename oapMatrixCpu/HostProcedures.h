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

#ifndef HOSTPROCEDURE_H
#define HOSTPROCEDURE_H

#include "Matrix.h"
#include "HostKernel.h"
#include "HostKernelExecutor.h"

#include "GenericProceduresApi.h"
#include "oapHostMatrixUtils.h"

class HostProcedures {
 public:
  HostProcedures(uint maxThreadsPerBlock = 1024);
  virtual ~HostProcedures();

  void setMaxThreadsPerBlock (uintt maxThreadsPerBlock);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

  bool isEqual(math::Matrix* matrix1, math::Matrix* matrix2);

  void substract(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

  void dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
  void dotProductShared (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);

  void dotProductPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);

  void dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, uintt dims[3][2], uintt periodicRows);

  void dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, uintt dims[3][2])
  {
    uintt periodicRows = oap::host::GetRows (matrix1);
    dotProductDimPeriodic (output, matrix1, matrix2, dims, periodicRows);
  }

  void dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, uintt dims[3][2]);

  void dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                   uintt outputDim[2], uintt params0Dim[2], uintt params1Dim[2])
  {
    uintt dims[3][2] = {{outputDim[0], outputDim[1]}, {params0Dim[0], params0Dim[1]}, {params1Dim[0], params1Dim[1]}};
    dotProduct (output, matrix1, matrix2, dims);
  }

  void transpose(math::Matrix* output, math::Matrix* matrix);

  void tanh (math::Matrix* output, math::Matrix* matrix);
  void sigmoid (math::Matrix* output, math::Matrix* matrix);
  void linear (math::Matrix* output, math::Matrix* matrix);
  void sin (math::Matrix* output, math::Matrix* matrix);
  void prelu (math::Matrix* output, math::Matrix* matrix);
  void relu (math::Matrix* output, math::Matrix* matrix);
  void softplus (math::Matrix* output, math::Matrix* matrix);

  void tanh (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void sigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void linear (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void sin (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void prelu (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void relu (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void softplus (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);

  void tanh (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void sigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void linear (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void sin (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void prelu (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void relu (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void softplus (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);

  void dprelu (math::Matrix* output, math::Matrix* matrix);
  void drelu (math::Matrix* output, math::Matrix* matrix);
  void dprelu (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void drelu (math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void dprelu (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  void drelu (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);

  void sum (floatt& reoutput, floatt& imoutput, math::Matrix* params0);

  void crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1);

  void tensorProduct (math::Matrix* matrix, math::Matrix* params0, math::Matrix* params1, uintt dims[3][2]);

  inline void tensorProduct (math::Matrix* matrix, math::Matrix* params0, math::Matrix* params1, uintt dims1[2], uintt dims2[2], uintt dims3[2])
  {
    uintt dims[3][2] = {{dims1[0], dims1[1]}, {dims2[0], dims2[1]}, {dims3[0], dims3[1]}};
    tensorProduct (matrix, params0, params1, dims);
  }

  void QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT);
  void setIdentity (math::Matrix* matrix);

  void setVector (math::Matrix* V, uintt column, math::Matrix* v, uintt length);

  void getVector (math::Matrix* vector, uintt length, math::Matrix* matrix, uintt column);

  void getVector (math::Matrix* vector, math::Matrix* matrix, uintt column);

  void convolve (math::Matrix* output, const math::Matrix* matrix, const math::Matrix* kernel);
  void poolAverage (math::Matrix* output, const math::Matrix* matrix, const math::MatrixDim& kernel);

 private:
  uint m_threads[2];
  uint m_blocks[2];

  uintt m_maxThreadsPerBlock;
  HostKernelExecutor m_kernel;

  void prepare(math::Matrix* matrix, HostKernel& hostKernel);
  void prepare(size_t w, size_t h, HostKernel& hostKernel);

  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> m_bmApi;

  uintt* createKernelArray (uintt* hostArray, size_t length)
  {
    return hostArray;
  }

  void _funcDim (const std::string& kname, math::Matrix* output, math::Matrix* matrix, uintt dims[2]);
  void _funcDimPeriodic (const std::string& kname, math::Matrix* output, math::Matrix* matrix, uintt dims[2][2]);
  std::function<uintt*(uintt*, uintt)> m_createKernelArray;
};

#endif  // HOSTCOMPAREPROCEDURE_H
