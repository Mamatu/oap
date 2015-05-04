/*
 * File:   MatrixProcedures.cpp
 * Author: mmatula
 *
 * Created on January 3, 2015, 8:37 PM
 */

#include "MatrixProcedures.h"
#include "DebugLogs.h"
#include "ThreadsMapper.h"
#include <math.h>

void CuMatrix::prepareDims(uintt w, uintt h) {
  uintt blocks[2];
  uintt threads[2];
  utils::mapper::SetThreadsBlocks(blocks, threads, w, h, m_maxThreadsPerBlock);
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
  return ::cuda::Kernel::Execute(functionName, params, m_kernel, m_image);
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
      m_compareOperationOutput(0) {
  m_pathes[0] = "liboglaMatrixCuda.cubin";
  m_pathes[2] = NULL;
  init();
  m_magniuteOutput = CudaUtils::AllocDeviceObj<floatt>(0);
}

void CuMatrix::init() {
  cuda::Context::Instance().init();
  m_image = ::cuda::Kernel::LoadImage(m_pathes);
  CUdevprop devprop;
  m_kernel.getDeviceProperties(devprop);
  m_maxThreadsPerBlock = devprop.maxThreadsPerBlock;
}

CuMatrix::~CuMatrix() {
  CudaUtils::FreeDeviceObj(m_magniuteOutput);
  cuda::Kernel::FreeImage(m_image);
  cuda::Context::Instance().destroy();
}

void CuMatrix::dotProduct(math::Matrix* output, math::Matrix* params0,
                          math::Matrix* params1) {
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  void* params[] = {&output, &params0, &params1};
  m_cuResult = execute("CUDAKernel_DotProduct", w, h, params, 0);
}

void CuMatrix::dotProductEx(math::Matrix* output, math::Matrix* params0,
                            math::Matrix* params1, MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &params1, &matrixEx};
  const uintt w = CudaUtils::GetColumns(matrixEx);
  const uintt h = CudaUtils::GetRows(matrixEx);
  m_cuResult = execute("CUDAKernel_DotProductEx", w, h, params, 0);
}

void CuMatrix::dotProductOpt(math::Matrix* output, math::Matrix* params0,
                             math::Matrix* params1) {
  void* params[] = {&output, &params0, &params1};
  bool isRe = CudaUtils::GetReValues(output) != NULL;
  bool isIm = CudaUtils::GetImValues(output) != NULL;
  const uintt ocolumns = CudaUtils::GetColumns(output);
  const uintt orows = CudaUtils::GetRows(output);
  const uintt p1rows = CudaUtils::GetRows(params0);
  const uintt p2columns = CudaUtils::GetColumns(params1);
  uintt size = (ocolumns * p1rows + orows * p2columns) * sizeof(floatt);
  if (isRe && isIm) {
    size = size * 2;
  }
  size = size + 3;
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
  size = size + 3;
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
                         math::Matrix* params1) {
  void* params[] = {&output, &params0, &params1};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  m_cuResult = execute("CUDAKernel_Substract", w, h, params, 0);
}

void CuMatrix::addMatrix(math::Matrix* output, math::Matrix* params0,
                         math::Matrix* params1) {
  void* params[] = {&output, &params0, &params1};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  m_cuResult = execute("CUDAKernel_Add", w, h, params, 0);
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

void CuMatrix::QR(math::Matrix* Q, math::Matrix* R, math::Matrix* H,
                  math::Matrix* aux0, math::Matrix* aux1, math::Matrix* aux2,
                  math::Matrix* aux3) {
  void* params[] = {&Q, &R, &H, &aux0, &aux1, &aux2, &aux3};
  const uintt w = CudaUtils::GetColumns(H);
  const uintt h = CudaUtils::GetRows(H);
  m_cuResult = execute("CUDAKernel_QR", w, h, params, 0);
}

bool CuMatrix::isUpperTriangular(math::Matrix* matrix) {
  void* params[] = {&matrix};
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

  m_cuResult = ::cuda::Kernel::Execute(cuKernelName, params, m_kernel, m_image);

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

  m_cuResult = ::cuda::Kernel::Execute(cuKernelName, params, m_kernel, m_image);

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

uintt CuMatrix::getCompareOperationSum() const {
  return m_compareOperationOutput;
}

CUresult CuMatrix::getStatus() const { return m_cuResult; }
