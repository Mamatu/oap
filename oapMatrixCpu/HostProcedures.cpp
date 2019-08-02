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

#include "HostProcedures.h"
#include "CuProcedures/CuCompareOptProcedures.h"
#include "CuProcedures/CuSubstractionProcedures.h"
#include "CuProcedures/CuDotProductProcedures.h"
#include "CuProcedures/CuTransposeProcedures.h"

#include "GenericProceduresApi.h"

#include "ThreadsMapper.h"

#include "HostBuffer.h"
#include "HostKernel.h"

#include "oapHostMatrixUtils.h"

#define CHECK_MATRIX(m) throwExceptionMsg (m != NULL, "Matrix is nullptr.");

class SubstractionImpl : public HostKernel {
 public:
  SubstractionImpl(math::Matrix* output, math::Matrix* param1,
                   math::Matrix* param2)
      : m_output(output), m_param1(param1), m_param2(param2) {}

  virtual ~SubstractionImpl() {}

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_substractMatrices(m_output, m_param1, m_param2);
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
    CUDA_dotProduct(m_output, m_param1, m_param2);
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
    T output = 0;
    for (uintt fa = 0; fa < length; ++fa) {
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

void HostProcedures::prepare(math::Matrix* matrix, HostKernel& hostKernel) {
  const uint columns = matrix->columns;
  const uint rows = matrix->rows;

  utils::mapper::SetThreadsBlocks(m_blocks, m_threads, columns, rows,
                                  m_threadsCount);

  hostKernel.setDims(m_blocks, m_threads);
}

HostProcedures::HostProcedures(uint maxThreadsPerBlock) : m_kernel(maxThreadsPerBlock), m_threadsCount(4) {}

HostProcedures::~HostProcedures() {}

void HostProcedures::setThreadsCount(uintt threadsCount) {
  m_threadsCount = threadsCount;
}

bool HostProcedures::compare(math::Matrix* matrix1, math::Matrix* matrix2) {
  if (matrix1->columns != matrix2->columns || matrix1->rows != matrix2->rows) {
    return false;
  }
  CompareImpl compareImpl(matrix1, matrix2);
  prepare(matrix1, compareImpl);
  compareImpl.executeKernelAsync();
  uintt sums = compareImpl.getSum();
  return sums == matrix1->rows * matrix1->columns;
}

bool HostProcedures::isEqual(math::Matrix* matrix1, math::Matrix* matrix2) {
  return compare(matrix1, matrix2);
}

void HostProcedures::substract(math::Matrix* output, math::Matrix* matrix1,
                               math::Matrix* matrix2) {
  SubstractionImpl substractionImpl(output, matrix1, matrix2);
  prepare(output, substractionImpl);
  substractionImpl.executeKernelAsync();
}

void HostProcedures::dotProduct(math::Matrix* output, math::Matrix* matrix1,
                                math::Matrix* matrix2) {
  DotProductImpl dotProductImpl(output, matrix1, matrix2);
  prepare(output, dotProductImpl);
  dotProductImpl.executeKernelAsync();
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
  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> bapi (oap::host::GetMatrixInfo);

  oap::generic::executeKernel1Arg ("CUDAKernel_Sigmoid", output, matrix, &m_kernel, bapi, true, [](){});
}

void HostProcedures::linear (math::Matrix* output, math::Matrix* matrix)
{
  oap::host::CopyHostMatrixToHostMatrix (output, matrix);
}

void HostProcedures::sin (math::Matrix* output, math::Matrix* matrix)
{
  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> bapi (oap::host::GetMatrixInfo);

  oap::generic::executeKernel1Arg ("CUDAKernel_Sin", output, matrix, &m_kernel, bapi, true, [](){});
}

void HostProcedures::sum (floatt& reoutput, floatt& imoutput, math::Matrix* params0)
{
  oap::host::HostBuffer<floatt> m_hsumsReBuffer;
  oap::host::HostBuffer<floatt> m_dsumsReBuffer;
  oap::host::HostBuffer<floatt> m_hsumsImBuffer;
  oap::host::HostBuffer<floatt> m_dsumsImBuffer;

  using GetAddressType = std::function<floatt*(const math::Matrix*)>;
  using GetAddressTypeRef = GetAddressType&;

  GetAddressType getReValues = [](const math::Matrix* matrix) -> floatt*
  {
    return matrix->reValues;
  };

  GetAddressType getImValues = [](const math::Matrix* matrix) -> floatt*
  {
    return matrix->imValues;
  };

  oap::generic::SumApi<decltype(oap::host::GetMatrixInfo), decltype(memcpy), GetAddressTypeRef>
  sumApi (oap::host::GetMatrixInfo, memcpy, getReValues, getImValues);

  oap::generic::SumBuffers<oap::host::HostBuffer<floatt>, oap::host::HostBuffer<floatt>>
  sumBuffers (m_hsumsReBuffer, m_dsumsReBuffer, m_hsumsImBuffer, m_dsumsImBuffer);

  oap::generic::sum (reoutput, imoutput, params0, &m_kernel, sumApi, sumBuffers);
}

void HostProcedures::crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> bapi (oap::host::GetMatrixInfo);

  oap::generic::crossEntropy (output, params0, params1, &m_kernel, bapi);
}
