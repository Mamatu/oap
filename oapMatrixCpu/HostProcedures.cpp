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

#include "HostProcedures.h"

#include <functional>

#include "CuProcedures/CuCompareOptProcedures.h"
#include "CuProcedures/CuSubtractionProcedures.h"
#include "CuProcedures/CuDotProductSpecificProcedures.h"
#include "CuProcedures/CuTransposeProcedures.h"

#include "GenericValidationApi.h"

#include "ThreadsMapper.h"

#include "HostBuffer.h"
#include "HostKernel.h"

namespace oap
{

//#define CHECK_MATRIX(m) throwExceptionMsg (m != NULL, "Matrix is nullptr.");

class SubtractionImpl : public HostKernel {
 public:
  SubtractionImpl(math::Matrix* output, math::Matrix* param1,
                   math::Matrix* param2)
      : m_output(output), m_param1(param1), m_param2(param2) {}

  virtual ~SubtractionImpl() {}

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_subtractMatrices(m_output, m_param1, m_param2);
  }

  virtual void onChange(HostKernel::ContextChange contextChnage,
                        const dim3& threadIdx, const dim3& blockIdx) {}

 private:
  math::Matrix* m_output;
  math::Matrix* m_param1;
  math::Matrix* m_param2;
};

class DotProductImpl : public HostKernel {
 public:
  DotProductImpl(math::Matrix* output, math::Matrix* param1,
                 math::Matrix* param2)
      : m_output(output), m_param1(param1), m_param2(param2) {}

  virtual ~DotProductImpl() {}

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_specific_dotProduct(m_output, m_param1, m_param2);
  }

  virtual void onChange(HostKernel::ContextChange contextChnage,
                        const dim3& threadIdx, const dim3& blockIdx) {}

 private:
  math::Matrix* m_output;
  math::Matrix* m_param1;
  math::Matrix* m_param2;
};

class CompareImpl : public HostKernel {
 public:
  CompareImpl(math::Matrix* param1, math::Matrix* param2)
      : m_param1(param1), m_param2(param2), m_buffer(NULL), m_sums(NULL) {}

  virtual ~CompareImpl() {
    delete[] m_sums;
    delete[] m_buffer;
  }

  uintt getSum() const { return getSum(m_sums, m_sumsLength); }

 protected:
  template <typename T>
  T getSum(T* buffer, size_t length) const {
    T output;
    for (uintt fa; fa < length; ++fa) {
      output += buffer[fa];
    }
    return output;
  }

  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_compareOpt(m_sums, m_param1, m_param2, m_buffer);
  }

  virtual void onChange(HostKernel::ContextChange contextChange,
                        const dim3& threadIdx, const dim3& blockIdx) {
    if (contextChange == HostKernel::CUDA_BLOCK) {
      // int actualSum = getSum(m_buffer, m_bufferLength);
      // m_sums[gridDim.x * blockIdx.y + blockIdx.x] = actualSum;
      memset(m_buffer, 0, sizeof(floatt) * m_bufferLength);
    }
  }

  virtual void onSetDims(const dim3& gridDim, const dim3& blockDim) {
    m_bufferLength = (blockDim.x * blockDim.y) / 2;
    m_sumsLength = gridDim.x * gridDim.y;
    m_buffer = new floatt[m_bufferLength];
    m_sums = new floatt[m_sumsLength];
    memset(m_buffer, 0, sizeof(floatt) * m_bufferLength);
    memset(m_sums, 0, sizeof(floatt) * m_sumsLength);
  }

 private:
  math::Matrix* m_param1;
  math::Matrix* m_param2;
  floatt* m_buffer;
  floatt* m_sums;
  size_t m_bufferLength;
  size_t m_sumsLength;
};

class TransposeImpl : public HostKernel {
 public:
  TransposeImpl(math::Matrix* output, math::Matrix* param)
      : m_output(output), m_param(param) {}

  virtual ~TransposeImpl() {}

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_transposeMatrix(m_output, m_param);
  }

  virtual void onChange(HostKernel::ContextChange contextChange,
                        const dim3& threadIdx, const dim3& blockIdx) {}

  virtual void onSetDims(const dim3& gridDim, const dim3& blockDim) {}

 private:
  math::Matrix* m_output;
  math::Matrix* m_param;
};

void HostProcedures::prepare (math::Matrix* matrix, HostKernel& hostKernel)
{
  const uint columns = gColumns (matrix);
  const uint rows = gRows (matrix);

  oap::utils::mapper::SetThreadsBlocks (m_blocks, m_threads, columns, rows, m_maxThreadsPerBlock);

  hostKernel.setDims(m_blocks, m_threads);
}

void HostProcedures::prepare(size_t w, size_t h, HostKernel& hostKernel) {
  const uint columns = w;
  const uint rows = h;

  oap::utils::mapper::SetThreadsBlocks(m_blocks, m_threads, columns, rows, m_maxThreadsPerBlock);

  hostKernel.setDims(m_blocks, m_threads);
}

HostProcedures::HostProcedures(uint maxThreadsPerBlock) : m_kernel(maxThreadsPerBlock), m_maxThreadsPerBlock (maxThreadsPerBlock), m_bmApi (oap::host::GetMatrixInfo),
m_createKernelArray(std::bind(&HostProcedures::createKernelArray, this, std::placeholders::_1, std::placeholders::_2))
{}

HostProcedures::~HostProcedures() {}

void HostProcedures::setMaxThreadsPerBlock (uintt threadsPerBlock)
{
  m_kernel.setMaxThreadsPerBlock (threadsPerBlock);
  m_maxThreadsPerBlock = threadsPerBlock;
}

bool HostProcedures::compare(math::Matrix* matrix1, math::Matrix* matrix2) {
  if (gColumns (matrix1) != gColumns (matrix2) || gRows (matrix1) != gRows (matrix2)) {
    return false;
  }
  CompareImpl compareImpl(matrix1, matrix2);
  prepare(matrix1, compareImpl);
  compareImpl.executeKernelAsync();
  uintt sums = compareImpl.getSum();
  return sums == gRows (matrix1) * gColumns (matrix1);
}

bool HostProcedures::isEqual(math::Matrix* matrix1, math::Matrix* matrix2) {
  return compare(matrix1, matrix2);
}

void HostProcedures::subtract(math::Matrix* output, math::Matrix* matrix1,
                               math::Matrix* matrix2) {
  SubtractionImpl subtractionImpl(output, matrix1, matrix2);
  prepare(output, subtractionImpl);
  subtractionImpl.executeKernelAsync();
}

void HostProcedures::dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2)
{
  oap::generic::dotProduct (output, matrix1, matrix2, &m_kernel, m_bmApi, [](){});
}

void HostProcedures::dotProductShared (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2)
{
  oap::generic::dotProductShared (output, matrix1, matrix2, &m_kernel, m_bmApi, [](){});
}

void HostProcedures::dotProductPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2)
{
  oap::generic::dotProductPeriodic (output, matrix1, matrix2, &m_kernel, m_bmApi, [](){},
                  m_createKernelArray);
}

void HostProcedures::dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, oap::generic::Dim32 dim, uintt periodicRows)
{
  oap::generic::dotProductDimPeriodic (output, matrix1, matrix2, dim, periodicRows, &m_kernel, m_bmApi, [](){}, m_createKernelArray);
}

void HostProcedures::dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, oap::generic::Dim32 dim)
{
  oap::generic::dotProduct (output, matrix1, matrix2, dim, &m_kernel, m_bmApi, [](){}, m_createKernelArray);
}

void HostProcedures::transpose(math::Matrix* output, math::Matrix* matrix) {
  TransposeImpl transposeImpl(output, matrix);
  prepare(output, transposeImpl);
  transposeImpl.executeKernelAsync();
}

void HostProcedures::tanh(math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> bapi (oap::host::GetMatrixInfo);

  oap::generic::executeKernel1Arg ("CUDAKernel_Tanh", output, matrix, &m_kernel, bapi, true, [](){});
}

void HostProcedures::sigmoid (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_Sigmoid", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::linear (math::Matrix* output, math::Matrix* matrix)
{
  oap::host::CopyHostMatrixToHostMatrix (output, matrix);
}

void HostProcedures::sin (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_Sin", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::prelu (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_PRelu", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::relu (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_Relu", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::softplus (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_Softplus", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dprelu (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DPRelu", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::drelu (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DRelu", output, matrix, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dprelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DPReluDim", output, matrix, dim);
}

void HostProcedures::drelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DReluDim", output, matrix, dim);
}

void HostProcedures::dprelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DPReluDimPeriodic", output, matrix, dim);
}

void HostProcedures::drelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DReluDimPeriodic", output, matrix, dim);
}

void HostProcedures::_funcDim (const std::string& kname, math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  oap::generic::executeKernel1Arg (kname, output, matrix, dim, &m_kernel, m_bmApi, true, [](){},
                                  m_createKernelArray);
}

void HostProcedures::tanh(math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_TanhDim", output, matrix, dim);
}

void HostProcedures::sigmoid (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_SigmoidDim", output, matrix, dim);
}

void HostProcedures::linear (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  debugAssert ("Not supported yet" == nullptr);
}

void HostProcedures::sin (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_SinDim", output, matrix, dim);
}

void HostProcedures::prelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_PReluDim", output, matrix, dim);
}

void HostProcedures::relu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_ReluDim", output, matrix, dim);
}

void HostProcedures::softplus (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_SoftplusDim", output, matrix, dim);
}

void HostProcedures::_funcDimPeriodic (const std::string& kname, math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  oap::generic::funcDimPeriodic (kname, output, matrix, dim, &m_kernel, m_bmApi, [](){},
                                  m_createKernelArray);
}

void HostProcedures::tanh (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_TanhDimPeriodic", output, matrix, dim);
}

void HostProcedures::sigmoid (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_SigmoidDimPeriodic", output, matrix, dim);
}

void HostProcedures::linear (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  debugAssert ("Not supported yet" == nullptr);
  //_funcDimPeriodic ("CUDAKernel_LinearDimPeriodic", output, matrix, dim);
}

void HostProcedures::sin (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_SinDimPeriodic", output, matrix, dim);
}

void HostProcedures::prelu(math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_PReluDimPeriodic", output, matrix, dim);
}

void HostProcedures::relu(math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_ReluDimPeriodic", output, matrix, dim);
}

void HostProcedures::softplus (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_SoftplusDimPeriodic", output, matrix, dim);
}

void HostProcedures::crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> bapi (oap::host::GetMatrixInfo);

  oap::generic::crossEntropy (output, params0, params1, &m_kernel, bapi);
}

void HostProcedures::tensorProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, oap::generic::Dim32 dim)
{
  oap::generic::tensorProduct (output, matrix1, matrix2, dim, &m_kernel, m_bmApi, [](){}, m_createKernelArray);
}

void HostProcedures::QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT)
{
  oap::generic::qrDecomposition_HT (Q, R, A, V, VT, P, VVT, &m_kernel, *this, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::setIdentity (math::Matrix* matrix)
{
  oap::generic::setIdentityMatrix (matrix, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::setVector (math::Matrix* V, uintt column, math::Matrix* v, uintt length)
{
  oap::generic::setVector (V, column, v, length, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::getVector (math::Matrix* vector, uintt length, math::Matrix* matrix, uintt column)
{
  oap::generic::getVector (vector, length, matrix, column, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::getVector (math::Matrix* vector, math::Matrix* matrix, uintt column)
{
  oap::generic::getVector (vector, matrix, column, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::convolve (math::Matrix* output, const math::Matrix* param, const math::Matrix* kernel)
{
  oap::generic::convolve (output, param, kernel, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::poolAverage (math::Matrix* output, const math::Matrix* matrix, const math::MatrixDim& kernel)
{
  oap::generic::poolAverage (output, matrix, kernel, &m_kernel, oap::host::GetMatrixInfo, [](){}, m_createKernelArray);
}

void HostProcedures::dsigmoid (math::Matrix* output, math::Matrix* input)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DRelu", output, input, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dlinear (math::Matrix* output, math::Matrix* input)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DLinear", output, input, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dtanh (math::Matrix* output, math::Matrix* input)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DTanh", output, input, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dsin (math::Matrix* output, math::Matrix* input)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DSin", output, input, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dsoftplus (math::Matrix* output, math::Matrix* input)
{
  oap::generic::executeKernel1Arg ("CUDAKernel_DSoftplus", output, input, &m_kernel, m_bmApi, true, [](){});
}

void HostProcedures::dsigmoid (math::Matrix* output, math::Matrix* input, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DSigmoid", output, input, dim);
}

void HostProcedures::dlinear (math::Matrix* output, math::Matrix* input, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DLinear", output, input, dim);
}

void HostProcedures::dtanh (math::Matrix* output, math::Matrix* input, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DTanh", output, input, dim);
}

void HostProcedures::dsin (math::Matrix* output, math::Matrix* input, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DSin", output, input, dim);
}

void HostProcedures::dsoftplus (math::Matrix* output, math::Matrix* input, oap::generic::Dim2 dim)
{
  _funcDim ("CUDAKernel_DSoftplus", output, input, dim);
}

void HostProcedures::dsigmoid (math::Matrix* output, math::Matrix* input, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DSigmoid", output, input, dim);
}

void HostProcedures::dlinear (math::Matrix* output, math::Matrix* input, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DLinear", output, input, dim);
}

void HostProcedures::dtanh (math::Matrix* output, math::Matrix* input, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DTanh", output, input, dim);
}

void HostProcedures::dsin (math::Matrix* output, math::Matrix* input, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DSin", output, input, dim);
}

void HostProcedures::dsoftplus (math::Matrix* output, math::Matrix* input, oap::generic::Dim22 dim)
{
  _funcDimPeriodic ("CUDAKernel_DSoftplus", output, input, dim);
}

void HostProcedures::hadamardProductVec (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  oap::generic::hadamardProductVec (output, param1, param2, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::add (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  oap::generic::add (output, param1, param2, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

void HostProcedures::multiplyReConstant (math::Matrix* output, math::Matrix* param1, floatt re)
{
  oap::generic::multiplyReConst (output, param1, re, &m_kernel, oap::host::GetMatrixInfo, [](){});
}

/*void HostProcedures::sum (floatt& reoutput, floatt& imoutput, math::Matrix* params0)
{
  oap::host::HostBuffer<floatt> m_hsumsReBuffer;
  oap::host::HostBuffer<floatt> m_dsumsReBuffer;
  oap::host::HostBuffer<floatt> m_hsumsImBuffer;
  oap::host::HostBuffer<floatt> m_dsumsImBuffer;

  using GetAddressType = std::function<floatt*(const math::Matrix*)>;
  using GetAddressTypeRef = GetAddressType&;

  GetAddressType getReValues = [](const math::Matrix* matrix) -> floatt*
  {
    return gReValues (matrix);
  };

  GetAddressType getImValues = [](const math::Matrix* matrix) -> floatt*
  {
    return gImValues (matrix);
  };

  oap::generic::SumApi<decltype(oap::host::GetMatrixInfo), decltype(memcpy), GetAddressTypeRef>
  sumApi (oap::host::GetMatrixInfo, memcpy, getReValues, getImValues);

  oap::generic::SumBuffers<oap::host::HostBuffer<floatt>, oap::host::HostBuffer<floatt>>
  sumBuffers (m_hsumsReBuffer, m_dsumsReBuffer, m_hsumsImBuffer, m_dsumsImBuffer);

  oap::generic::sum (reoutput, imoutput, params0, &m_kernel, sumApi, sumBuffers);
}*/

void HostProcedures::sum (floatt& reoutput, floatt& imoutput, const math::Matrix* param)
{
  //m_cuStatus = 
  oap::generic::sum (reoutput, imoutput, param, &m_kernel, oap::host::GetMatrixInfo, oap::host::GetRefHostMatrix, memcpy, m_hsumsReBuffer, m_hsumsImBuffer, m_dsumsReBuffer, m_dsumsImBuffer);
}

void HostProcedures::setZeroMatrix (math::Matrix* param)
{
  oap::host::SetZeroMatrix (param);
}
}
