/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "CuMatrixProcedures/CuCompareOptProcedures.h"
#include "CuMatrixProcedures/CuSubstractionProcedures.h"
#include "CuMatrixProcedures/CuDotProductProcedures.h"
#include "CuMatrixProcedures/CuTransposeProcedures.h"
#include "ThreadsMapper.h"
#include "HostKernel.h"

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
      memset(m_buffer, 0, sizeof(int) * m_bufferLength);
    }
  }

  virtual void onSetDims(const dim3& gridDim, const dim3& blockDim) {
    m_bufferLength = (blockDim.x * blockDim.y) / 2;
    m_sumsLength = gridDim.x * gridDim.y;
    m_buffer = new int[m_bufferLength];
    m_sums = new int[m_sumsLength];
    memset(m_buffer, 0, sizeof(int) * m_bufferLength);
    memset(m_sums, 0, sizeof(int) * m_sumsLength);
  }

 private:
  math::Matrix* m_param1;
  math::Matrix* m_param2;
  int* m_buffer;
  int* m_sums;
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

HostProcedures::HostProcedures() : m_threadsCount(4) {}

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
