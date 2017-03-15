/*
 * Copyright 2016 Marcin Matula
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

#include "MatrixProcedures.h"
#include "DebugLogs.h"
#include "ThreadsMapper.h"
#include "HostMatrixKernels.h"
#include <math.h>

void CuMatrix::prepareDims(uintt w, uintt h) {
  uintt blocks[2];
  uintt threads[2];
  m_kernel.calculateThreadsBlocks(blocks, threads, w, h);
  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);
}

CUresult CuMatrix::execute(const char* functionName, uintt w, uintt h,
                           void** params, uintt sharedMemory,
                           bool _prepareDims) {
  if (_prepareDims) {
    prepareDims(w, h);
  }
  m_kernel.setSharedMemory(sharedMemory);
  resetFlags();
  return ::device::Kernel::Execute(functionName, params, m_kernel);
}

CuMatrix::CuMatrix()
    : m_cuResult(CUDA_SUCCESS),
      m_dmagnitudeOutputBuffer(CuMatrix::CUDA),
      m_dmagnitudeBuffer(CuMatrix::CUDA),
      m_hmagnitudeOutputBuffer(CuMatrix::HOST),
      m_dcompareOutputBuffer(CuMatrix::CUDA),
      m_dcompareBuffer(CuMatrix::CUDA),
      m_hcompareOutputBuffer(CuMatrix::HOST),
      m_magnitudeBuffer(CuMatrix::CUDA),
      m_disuppertriangularOutputBuffer(CuMatrix::CUDA),
      m_hisuppertriangularOutputBuffer(CuMatrix::HOST),
      m_dqrSums(CuMatrix::CUDA),
      m_dqrBuffer(CuMatrix::CUDA),
      m_compareOperationOutput(0) {
  init();
  m_magniuteOutput = CudaUtils::AllocDeviceObj<floatt>(0);
  m_doutputIsTriangular = CudaUtils::AllocDeviceObj<int>(0);
}

void CuMatrix::init() {
  m_kernel.load("liboapMatrixCuda.cubin");
  CUdevprop devprop;
  m_kernel.getDeviceProperties(devprop);
  m_maxThreadsPerBlock = devprop.maxThreadsPerBlock;
}

CuMatrix::~CuMatrix() {
  CudaUtils::FreeDeviceObj(m_magniuteOutput);
  CudaUtils::FreeDeviceObj(m_doutputIsTriangular);
  m_kernel.unload();
}

void CuMatrix::dotProduct(math::Matrix* output, math::Matrix* params0,
                          math::Matrix* params1, uintt columns, uintt rows) {
  void* params[] = {&output, &params0, &params1};
  m_cuResult = execute("CUDAKernel_DotProduct", columns, rows, params, 0);
}

void CuMatrix::calculateQTHQ(math::Matrix* output, math::Matrix* H,
                             math::Matrix* Q, math::Matrix* aux) {
  transposeMatrix(output, Q);
  dotProduct(aux, H, output);
  dotProduct(output, Q, aux);
}

void CuMatrix::dotProductEx(math::Matrix* output, math::Matrix* params0,
                            math::Matrix* params1, MatrixEx* matrixEx,
                            uintt columns, uintt rows) {
  void* params[] = {&output, &params0, &params1, &matrixEx};
  m_cuResult = execute("CUDAKernel_DotProductEx", columns, rows, params, 0);
}

void CuMatrix::dotProductOpt(math::Matrix* output, math::Matrix* params0,
                             math::Matrix* params1, uintt ocolumns, uintt orows,
                             uintt p1rows, uintt p2columns) {
  void* params[] = {&output, &params0, &params1};
  bool isRe = CudaUtils::GetReValues(output) != NULL;
  bool isIm = CudaUtils::GetImValues(output) != NULL;
  uintt size = (ocolumns * p1rows + orows * p2columns) * sizeof(floatt);
  if (isRe && isIm) {
    size = size * 2;
  }
  size = size * 3;
  m_cuResult =
      execute("CUDAKernel_DotProductOpt", ocolumns, orows, params, size);
}

void CuMatrix::dotProductExOpt(math::Matrix* output, math::Matrix* params0,
                               math::Matrix* params1, MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &params1, &matrixEx};
  bool isRe = CudaUtils::GetReValues(output) != NULL;
  bool isIm = CudaUtils::GetImValues(output) != NULL;
  const uintt ocolumns = CudaUtils::GetColumns(matrixEx);
  const uintt orows = CudaUtils::GetRows(matrixEx);
  const uintt p1rows = CudaUtils::GetRows(params0);
  const uintt p2columns = CudaUtils::GetColumns(params1);
  uintt size = (ocolumns * p1rows + orows * p2columns) * sizeof(floatt);
  if (isRe && isIm) {
    size = size * 2;
  }
  size = size * 3;
  m_cuResult =
      execute("CUDAKernel_DotProductExOpt", ocolumns, orows, params, size);
}

void CuMatrix::transposeMatrixEx(math::Matrix* output, math::Matrix* params0,
                                 MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &matrixEx};
  const uintt w = CudaUtils::GetColumns(matrixEx);
  const uintt h = CudaUtils::GetRows(matrixEx);
  execute("CUDAKernel_TransposeEx", w, h, params, 0);
}

void CuMatrix::transposeMatrix(math::Matrix* output, math::Matrix* params0) {
  void* params[] = {&output, &params0};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  m_cuResult = execute("CUDAKernel_Transpose", w, h, params, 0);
}

void CuMatrix::substract(math::Matrix* output, math::Matrix* params0,
                         math::Matrix* params1, uintt columns, uintt rows) {
  void* params[] = {&output, &params0, &params1};
  m_cuResult = execute("CUDAKernel_Substract", columns, rows, params, 0);
}

void CuMatrix::add(math::Matrix* output, math::Matrix* params0,
                   math::Matrix* params1, uintt columns, uintt rows) {
  void* params[] = {&output, &params0, &params1};
  m_cuResult = execute("CUDAKernel_Add", columns, rows, params, 0);
}

void CuMatrix::setVector(math::Matrix* V, uintt column, math::Matrix* v,
                         uintt length) {
  const uintt w = CudaUtils::GetColumns(v);
  const uintt h = CudaUtils::GetRows(v);
  void* params[] = {&V, &column, &v, &length};
  m_cuResult = execute("CUDAKernel_SetVector", w, h, params, 0);
}

void CuMatrix::getVector(math::Matrix* vector, uintt length,
                         math::Matrix* matrix, uintt column) {
  const uintt w = CudaUtils::GetColumns(vector);
  const uintt h = CudaUtils::GetRows(vector);
  void* params[] = {&vector, &length, &matrix, &column};
  m_cuResult = execute("CUDAKernel_GetVector", w, h, params, 0);
}

void CuMatrix::getVector(math::Matrix* vector, math::Matrix* matrix,
                         uintt column) {
  const uintt w = CudaUtils::GetColumns(vector);
  const uintt h = CudaUtils::GetRows(vector);
  uintt length = w * h;
  void* params[] = {&vector, &length, &matrix, &column};
  m_cuResult = execute("CUDAKernel_GetVector", w, h, params, 0);
}

void CuMatrix::magnitude(floatt& output, math::Matrix* param0) {
  magnitude2(output, param0);
  output = sqrt(output);
}

void CuMatrix::magnitudeOpt(floatt& output, math::Matrix* param0) {
  magnitude2Opt(output, param0);
  output = sqrt(output);
}

void CuMatrix::magnitudeOptVer2(floatt& output, math::Matrix* param0) {
  magnitude2OptVer2(output, param0);
  output = sqrt(output);
}

void CuMatrix::magnitude2(floatt& output, math::Matrix* param0) {
  magnitude2Opt(output, param0);
}

void CuMatrix::magnitude2Opt(floatt& output, math::Matrix* params0) {
  const uintt w = CudaUtils::GetColumns(params0);
  const uintt h = CudaUtils::GetRows(params0);
  output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
}

void CuMatrix::magnitude2OptVer2(floatt& output, math::Matrix* params0) {
  const uintt w = CudaUtils::GetColumns(params0);
  const uintt h = CudaUtils::GetRows(params0);
  if (w > 1) {
    output =
        magnitude2Procedure("CUDAKernel_MagnitudeOptVer2", params0, w / 2, h);
  } else {
    output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
  }
}

void CuMatrix::setDiagonal(math::Matrix* matrix, floatt re, floatt im) {
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);
  void* params[] = {&matrix, &re, &im};
  m_cuResult = execute("CUDAKernel_SetDiagonal", w, h, params, 0);
}

void CuMatrix::setIdentity(math::Matrix* matrix) {
  void* params[] = {&matrix};
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);
  m_cuResult = execute("CUDAKernel_SetIdentity", w, h, params, 0);
}

void CuMatrix::setZeroMatrix(math::Matrix* matrix) {
  CudaUtils::SetZeroMatrix(matrix, true, true);
  m_cuResult = CUDA_SUCCESS;
}

void CuMatrix::QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* H,
                    math::Matrix* aux0, math::Matrix* aux1, math::Matrix* aux2,
                    math::Matrix* aux3) {
  void* params[] = {&Q, &R, &H, &aux0, &aux1, &aux2, &aux3};
  uint maxThreads = m_kernel.getMaxThreadsPerBlock();
  const uintt w = CudaUtils::GetColumns(H);
  const uintt h = CudaUtils::GetRows(H);
  if (maxThreads >= w * h) {
    m_cuResult = execute("CUDAKernel_QRGR", w, h, params, 0);
  } else {
    m_cuResult = HOSTKernel_QRGR(Q, R, H, aux0, aux1, aux2, aux3, m_kernel);
  }
}

bool CuMatrix::isUpperTriangular(math::Matrix* matrix) {
  int result = -10;
  void* params[] = {&m_doutputIsTriangular, &matrix};
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);
  m_cuResult = execute("CUDAKernel_IsUpperTriangular", w, h, params, 0);
  CudaUtils::CopyDeviceToHost(&result, m_doutputIsTriangular, sizeof(int));
  return result == 1;
}

void CuMatrix::multiplyConstantMatrix(math::Matrix* output,
                                      math::Matrix* params0, floatt re) {
  void* params[] = {&output, &params0, &re};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  m_cuResult = execute("CUDAKernel_MultiplyConstantRe", w, h, params, 0);
}

void CuMatrix::multiplyConstantMatrix(math::Matrix* output,
                                      math::Matrix* params0, floatt re,
                                      floatt im) {
  void* params[] = {&output, &params0, &re, &im};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  m_cuResult = execute("CUDAKernel_MultiplyConstant", w, h, params, 0);
}

bool CuMatrix::compare(math::Matrix* matrix1, math::Matrix* matrix2) {
  if (matrix1 == matrix2) {
    return true;
  }

  const uintt w = CudaUtils::GetColumns(matrix1);
  const uintt h = CudaUtils::GetRows(matrix1);

  return compareProcedure("CUDAKernel_CompareOpt", matrix1, matrix2, w, h, w,
                          h);
}

bool CuMatrix::compareVer2(math::Matrix* matrix1, math::Matrix* matrix2) {
  if (matrix1 == matrix2) {
    return true;
  }

  const uintt w = CudaUtils::GetColumns(matrix1);
  const uintt h = CudaUtils::GetRows(matrix1);

  return compareProcedure("CUDAKernel_CompareOptVer2", matrix1, matrix2, w, h,
                          w / 2, h);
}

bool CuMatrix::compareProcedure(const char* cuKernelName, math::Matrix* matrix1,
                                math::Matrix* matrix2, uintt w, uintt h,
                                uintt wthreads, uintt hthreads) {
  if (matrix1 == matrix2) {
    return true;
  }

  uintt blocks[2];
  uintt threads[2];

  m_kernel.calculateThreadsBlocks(blocks, threads, wthreads, hthreads);

  assert(threads[0] * threads[1] * sizeof(int) <
         m_kernel.getSharedMemorySize());

  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);
  m_kernel.setSharedMemory(threads[0] * threads[1] * sizeof(int));

  uintt outputLength = blocks[0] * blocks[1];

  m_dcompareOutputBuffer.realloc(outputLength);
  m_hcompareOutputBuffer.realloc(outputLength);

  void* params[] = {&m_dcompareOutputBuffer.m_buffer, &matrix1, &matrix2};

  m_cuResult = ::device::Kernel::Execute(cuKernelName, params, m_kernel);

  CudaUtils::CopyDeviceToHost(
      m_hcompareOutputBuffer.m_buffer, m_dcompareOutputBuffer.m_buffer,
      outputLength * m_hcompareOutputBuffer.GetSizeOfType());

  uintt outcome = 0;
  for (uint fa = 0; fa < blocks[0] * blocks[1]; ++fa) {
    outcome += m_hcompareOutputBuffer.m_buffer[fa];
  }

  m_compareOperationOutput = outcome;
  return outcome == w * h;
}

floatt CuMatrix::magnitude2Procedure(const char* cuKernelName,
                                     math::Matrix* matrix, uintt wthreads,
                                     uintt hthreads) {
  uintt blocks[2];
  uintt threads[2];

  m_kernel.calculateThreadsBlocks(blocks, threads, wthreads, hthreads);

  assert(threads[0] * threads[1] * sizeof(floatt) <
         m_kernel.getSharedMemorySize());

  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);
  m_kernel.setSharedMemory(threads[0] * threads[1] * sizeof(floatt));

  uintt outputLength = blocks[0] * blocks[1];

  m_dmagnitudeOutputBuffer.realloc(outputLength);
  m_hmagnitudeOutputBuffer.realloc(outputLength);

  void* params[] = {&m_dmagnitudeOutputBuffer.m_buffer, &matrix};

  m_cuResult = ::device::Kernel::Execute(cuKernelName, params, m_kernel);

  return magnitude2Procedure_GetOutput(blocks, outputLength);
}

floatt CuMatrix::magnitude2Procedure_GetOutput(uintt blocks[2],
                                               uintt outputLength) const {
  CudaUtils::CopyDeviceToHost(
      m_hmagnitudeOutputBuffer.m_buffer, m_dmagnitudeOutputBuffer.m_buffer,
      outputLength * m_dmagnitudeOutputBuffer.GetSizeOfType());

  floatt outcome = 0;
  for (uint fa = 0; fa < blocks[0] * blocks[1]; ++fa) {
    outcome += m_hmagnitudeOutputBuffer.m_buffer[fa];
  }

  return outcome;
}

void CuMatrix::qrProcedure(QRType qrType, math::Matrix* Q, math::Matrix* R,
                           math::Matrix* A, math::Matrix* AT, math::Matrix* P,
                           math::Matrix* I, math::Matrix* v, math::Matrix* vt,
                           math::Matrix* vvt) {
  uintt blocks[2];
  uintt threads[2];
  const uintt w = CudaUtils::GetColumns(A);
  const uintt h = CudaUtils::GetRows(A);

  m_kernel.calculateThreadsBlocks(blocks, threads, w, h);
  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);

  m_dqrSums.realloc(blocks[0] * blocks[1]);

  if (qrType == OPT) {
    m_kernel.setSharedMemory(h * sizeof(floatt));
    void* params[] = {&Q, &R, &A, &AT, &m_dqrSums.m_buffer, &P, &I, &v, &vt,
                      &vvt};
    m_cuResult =
        ::device::Kernel::Execute("CUDAKernel_QRHTOpt", params, m_kernel);
  } else {
    m_dqrBuffer.realloc(h);
    void* params[] = {&Q, &R, &A, &AT, &m_dqrSums.m_buffer,
                      &m_dqrBuffer.m_buffer, &P, &I, &v, &vt, &vvt};
    m_cuResult = ::device::Kernel::Execute("CUDAKernel_QRHT", params, m_kernel);
  }
}

uintt CuMatrix::getCompareOperationSum() const {
  return m_compareOperationOutput;
}

CUresult CuMatrix::getStatus() const { return m_cuResult; }
