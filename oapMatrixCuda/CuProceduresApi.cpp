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

#include "CuProceduresApi.h"

#include <functional>
#include <iterator>
#include <math.h>

#include "Logger.h"
#include "HostMatrixKernels.h"

#include "oapDeviceMatrixUPtr.h"
#include "oapDeviceMatrixPtr.h"
#include "oapHostMatrixUPtr.h"
#include "oapHostMatrixPtr.h"

#include "ThreadsMapper.h"
#include "oapCudaMatrixUtils.h"

#include "CudaCoreApi.h"
#include "Logger.h"

namespace oap
{

void CuProceduresApi::prepareDims (uintt w, uintt h)
{
  m_kernel.calculateThreadsBlocks(m_blocks, m_threads, w, h);
  m_kernel.setBlocksCount(m_blocks[0], m_blocks[1]);
  m_kernel.setThreadsCount(m_threads[0], m_threads[1]);
}

bool CuProceduresApi::execute (const char* functionName, uintt w, uintt h, void** params, uintt sharedMemory, bool _prepareDims)
{
  if (_prepareDims)
  {
    prepareDims(w, h);
  }
  m_kernel.setSharedMemory(sharedMemory);

  resetFlags();

  return ::oap::cuda::Kernel::Execute(functionName, const_cast<const void**>(params), m_kernel);
}

CuProceduresApi::CuProceduresApi()
    : m_cuStatus(CUDA_SUCCESS),
      m_compareOperationOutput(0),
      m_bmApi (oap::cuda::GetMatrixInfo),
      m_preExecCallback ([this](){ this->resetFlags ();}),
      m_createKernelArray ([this](uintt* hostArray, size_t length) { return this->createKernelArray (hostArray, length); })
{
  init();
  m_magnitudeOutput = CudaUtils::AllocDeviceObj<floatt>(0);
  m_doutputIsTriangular = CudaUtils::AllocDeviceObj<int>(0);
}

void CuProceduresApi::init() {
  m_kernel.load("liboapMatrixCuda.cubin");
  m_maxThreadsPerBlock = m_kernel.getMaxThreadsPerBlock ();
}

CuProceduresApi::~CuProceduresApi() {
  CudaUtils::FreeDeviceObj(m_magnitudeOutput);
  CudaUtils::FreeDeviceObj(m_doutputIsTriangular);
  oap::cuda::DeleteDeviceMatrixEx (m_dMatrixEx);

  deallocKernelArrays ();

  m_kernel.unload();
}

void CuProceduresApi::dotProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

#ifndef OAP_DOT_PRODUCT_SHARED_DEFAULT
  oap::generic::dotProduct (output, params0, params1, &m_kernel, m_bmApi, m_preExecCallback);
#else
  oap::generic::dotProductShared (output, params0, params1, &m_kernel, m_bmApi, m_preExecCallback);
#endif
}

void CuProceduresApi::dotProductShared (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::dotProductShared (output, params0, params1, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::addDotProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::check_dotProduct (output, params0, params1, m_bmApi);

  const void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_AddDotProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::tensorProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::check_tensorProduct (output, params0, params1, columns, rows, m_bmApi);

  const void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_TensorProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::tensorProduct (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, generic::Dim32 dim)
{
  m_cuStatus = oap::generic::tensorProduct (output, matrix1, matrix2, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::hadamardProduct (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  m_cuStatus = oap::generic::hadamardProduct (output, params0, params1, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::hadamardProductVec (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  m_cuStatus = oap::generic::hadamardProductVec (output, params0, params1, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::calculateQTHQ (math::ComplexMatrix* output, math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* aux)
{
  transpose(output, Q);
  dotProduct(aux, H, output);
  dotProduct(output, Q, aux);
}

void CuProceduresApi::dotProductPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2)
{
  m_cuStatus = oap::generic::dotProductPeriodic (output, matrix1, matrix2, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dotProductDimPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, generic::Dim32 dim, uintt periodicRows)
{
  m_cuStatus = oap::generic::dotProductDimPeriodic (output, matrix1, matrix2, dim, periodicRows, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dotProduct(math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                                 generic::Dim32 dim)
{
  m_cuStatus = oap::generic::dotProduct (output, matrix1, matrix2, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dotProductOpt(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                             math::ComplexMatrix* params1, uintt ocolumns, uintt orows,
                             uintt p1rows, uintt p2columns) {
  void* params[] = {&output, &params0, &params1};


  auto minfo = oap::cuda::GetMatrixInfo (output);
  const bool isRe = minfo.isRe;
  const bool isIm = minfo.isIm;

  uintt size = (ocolumns * p1rows + orows * p2columns) * sizeof(floatt);
  if (isRe && isIm) {
    size = size * 2;
  }
  size = size * 3;
  m_cuStatus =
      execute("CUDAKernel_DotProductOpt", ocolumns, orows, params, size);
}

void CuProceduresApi::dotProductExOpt(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                               math::ComplexMatrix* params1, MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &params1, &matrixEx};

  auto minfo = oap::cuda::GetMatrixInfo (output);
  const bool isRe = minfo.isRe;
  const bool isIm = minfo.isIm;

  const uintt ocolumns = CudaUtils::GetColumns(matrixEx);
  const uintt orows = CudaUtils::GetRows(matrixEx);
  const uintt p1rows = oap::cuda::GetRows(params0);
  const uintt p2columns = oap::cuda::GetColumns(params1);
  uintt size = (ocolumns * p1rows + orows * p2columns) * sizeof(floatt);
  if (isRe && isIm) {
    size = size * 2;
  }
  size = size * 3;
  m_cuStatus =
      execute("CUDAKernel_DotProductExOpt", ocolumns, orows, params, size);
}

void CuProceduresApi::transposeEx(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                 MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &matrixEx};
  const uintt w = CudaUtils::GetColumns(matrixEx);
  const uintt h = CudaUtils::GetRows(matrixEx);
  execute("CUDAKernel_TransposeEx", w, h, params, 0);
}

void CuProceduresApi::transpose(math::ComplexMatrix* output, math::ComplexMatrix* params0) {
  const uintt wo = oap::cuda::GetColumns (output);
  const uintt ho = oap::cuda::GetRows (output);

  const uintt wp = oap::cuda::GetColumns (params0);
  const uintt hp = oap::cuda::GetRows (params0);

  debugAssert (ho == wp && hp == wo);

  if ((wo == 1 && ho == wp && hp == 1) || (ho == 1 && hp == wo && wp == 1))
  {
    oap::cuda::CopyDeviceToDevice (output, params0);
  }
  else
  {
    void* params[] = {&output, &params0};
    m_cuStatus = execute("CUDAKernel_Transpose", wo, ho, params, 0);
  }
}

void CuProceduresApi::conjugateTranspose(math::ComplexMatrix* output, math::ComplexMatrix* params0) {

  const uintt wo = oap::cuda::GetColumns(output);
  const uintt ho = oap::cuda::GetRows(output);

  const uintt wp = oap::cuda::GetColumns (params0);
  const uintt hp = oap::cuda::GetRows (params0);

  debugAssert (ho == wp || hp == wo);

  if ((wo == 1 && ho == wp && hp == 1) || (ho == 1 && hp == wo && wp == 1))
  {
    oap::cuda::CopyDeviceToDevice(output, params0);
  }
  else
  {
    void* params[] = {&output, &params0};
    m_cuStatus = execute("CUDAKernel_ConjugateTranspose", wo, ho, params, 0);
  }
}

void CuProceduresApi::subtract(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows)
{
  void* params[] = {&output, &params0, &params1};
  m_cuStatus = execute("CUDAKernel_Substract", columns, rows, params, 0);
}

void CuProceduresApi::addSubstract(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows)
{
  void* params[] = {&output, &params0, &params1};
  m_cuStatus = execute("CUDAKernel_AddSubstract", columns, rows, params, 0);
}

void CuProceduresApi::add (math::ComplexMatrix* output, math::ComplexMatrix* param1, math::ComplexMatrix* param2, uintt columns, uintt rows)
{
  oap::generic::add (output, param1, param2, &m_kernel, oap::cuda::GetMatrixInfo, [](){});
}

void CuProceduresApi::add (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt value)
{
  auto minfo = oap::cuda::GetMatrixInfo (output);
  void* params[] = {&output, &params0, &value};
  m_cuStatus = execute ("CUDAKernel_AddMatrixValue", minfo.columns(), minfo.rows(), params, 0);
}

void CuProceduresApi::setVector (math::ComplexMatrix* V, uintt column, math::ComplexMatrix* v, uintt length)
{
  m_cuStatus = oap::generic::setVector (V, column, v, length, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::getVector (math::ComplexMatrix* vector, uintt length, math::ComplexMatrix* matrix, uintt column)
{
  m_cuStatus = oap::generic::getVector (vector, length, matrix, column, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::getVector (math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uintt column)
{
  m_cuStatus = oap::generic::getVector (vector, matrix, column, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::magnitude(floatt& output, math::ComplexMatrix* param0) {
  magnitude2(output, param0);
  output = sqrt(output);
}
/*
void CuProceduresApi::sum (floatt& output, math::ComplexMatrix* matrix)
{
  const uintt w = oap::cuda::GetColumns (matrix);
  const uintt h = oap::cuda::GetRows (matrix);
  void* params[] = {&output, &matrix};

  prepareDims (w, h);

  output = execute ("CUDAKernel_Sum", w, h, params, 0, false);
}
*/
void CuProceduresApi::sum (floatt& reoutput, floatt& imoutput, const math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::sum (reoutput, imoutput, matrix, &m_kernel, oap::cuda::GetMatrixInfo, oap::cuda::GetRefHostMatrix, CudaUtils::CopyDeviceToHost, m_hsumsReBuffer, m_hsumsImBuffer, m_dsumsReBuffer, m_dsumsImBuffer);
}

/*void CuProceduresApi::sum (floatt& reoutput, const math::ComplexMatrix* matrix)
{
  floatt imoutput;
  sum (reoutput, imoutput, matrix);
}
*/
#if 0
void CuProceduresApi::sum (floatt& reoutput, const floatt* values, size_t count)
{
  math::ComplexMatrix matrix;
  matrix.reValues = const_cast<floatt*>(values);
  matrix.columns = count;
  matrix.rows = 1;
  sum (reoutput, &matrix);
}
#endif

void CuProceduresApi::magnitudeOpt(floatt& output, math::ComplexMatrix* param0) {
  magnitude2Opt(output, param0);
  output = sqrt(output);
}

void CuProceduresApi::magnitudeOptVer2(floatt& output, math::ComplexMatrix* param0) {
  magnitude2OptVer2(output, param0);
  output = sqrt(output);
}

void CuProceduresApi::magnitude2(floatt& output, math::ComplexMatrix* param0) {
  magnitude2Opt(output, param0);
}

void CuProceduresApi::magnitude2Opt(floatt& output, math::ComplexMatrix* params0) {
  const uintt w = oap::cuda::GetColumns(params0);
  const uintt h = oap::cuda::GetRows(params0);
  output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
}

void CuProceduresApi::magnitude2OptVer2(floatt& output, math::ComplexMatrix* params0) {
  const uintt w = oap::cuda::GetColumns(params0);
  const uintt h = oap::cuda::GetRows(params0);
  if (w > 1) {
    output =
        magnitude2Procedure("CUDAKernel_MagnitudeOptVer2", params0, w / 2, h);
  } else {
    output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
  }
}

void CuProceduresApi::setDiagonal(math::ComplexMatrix* matrix, floatt re, floatt im) {
  const uintt w = oap::cuda::GetColumns(matrix);
  const uintt h = oap::cuda::GetRows(matrix);
  void* params[] = {&matrix, &re, &im};
  m_cuStatus = execute("CUDAKernel_SetDiagonal", w, h, params, 0);
}

void CuProceduresApi::setIdentity (math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::setIdentityMatrix (matrix, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::setZeroMatrix (math::ComplexMatrix* matrix)
{
  oap::cuda::SetZeroMatrix (matrix);
  m_cuStatus = CUDA_SUCCESS;
}

void CuProceduresApi::QRGR(math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* H,
                    math::ComplexMatrix* aux0, math::ComplexMatrix* aux1, math::ComplexMatrix* aux2,
                    math::ComplexMatrix* aux3) {
  void* params[] = {&Q, &R, &H, &aux0, &aux1, &aux2, &aux3};
  uint maxThreads = m_kernel.getMaxThreadsPerBlock();
  const uintt w = oap::cuda::GetColumns(H);
  const uintt h = oap::cuda::GetRows(H);
  if (maxThreads >= w * h)
  {
    m_cuStatus = execute("CUDAKernel_QRGR", w, h, params, 0);
  }
  else
  {
    oap::cuda::CudaMatrixApi cuMatrixApi;
    m_cuStatus = HOSTKernel_QRGR (Q, R, H, aux0, aux1, aux2, aux3, *this, cuMatrixApi, oap::cuda::CopyDeviceMatrixToDeviceMatrix);
  }
}

void CuProceduresApi::QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, math::ComplexMatrix* P, math::ComplexMatrix* VVT)
{
  m_cuStatus = oap::generic::qrDecomposition_HT (Q, R, A, V, VT, P, VVT, &m_kernel, *this, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

bool CuProceduresApi::isUpperTriangular(math::ComplexMatrix* matrix) {
  int result = -10;
  void* params[] = {&m_doutputIsTriangular, &matrix};
  const uintt w = oap::cuda::GetColumns(matrix);
  const uintt h = oap::cuda::GetRows(matrix);
  m_cuStatus = execute("CUDAKernel_IsUpperTriangular", w, h, params, 0);
  CudaUtils::CopyDeviceToHost(&result, m_doutputIsTriangular, sizeof(int));
  return result == 1;
}

void CuProceduresApi::calcTriangularH (math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* R,
                                       math::ComplexMatrix* aux1, math::ComplexMatrix* aux2, math::ComplexMatrix* aux3,
                                       math::ComplexMatrix* aux4, math::ComplexMatrix* aux5, math::ComplexMatrix* aux6)
{
  void* params[] = {&H, &Q, &R, &aux1, &aux2, &aux3, &aux4, &aux5, &aux6};
  const uintt w = oap::cuda::GetColumns (H);
  const uintt h = oap::cuda::GetRows (H);
  m_cuStatus = execute("CUDAKernel_CalculateTriangularH", w, h, params, 0);
}

void CuProceduresApi::calcTriangularHStep (math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* R,
                                           math::ComplexMatrix* aux1, math::ComplexMatrix* aux2, math::ComplexMatrix* aux3,
                                           math::ComplexMatrix* aux4, math::ComplexMatrix* aux5, math::ComplexMatrix* aux6)
{
  void* params[] = {&H, &Q, &R, &aux1, &aux2, &aux3, &aux4, &aux5, &aux6};
  const uintt w = oap::cuda::GetColumns (H);
  const uintt h = oap::cuda::GetRows (H);
  m_cuStatus = execute("CUDAKernel_CalculateTriangularHStep", w, h, params, 0);
}

void CuProceduresApi::multiplyReConstant(math::ComplexMatrix* output, math::ComplexMatrix* param1, floatt re)
{
  oap::generic::multiplyReConst (output, param1, re, &m_kernel, oap::cuda::GetMatrixInfo, [](){});
}

void CuProceduresApi::multiplyConstant(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                             floatt re, floatt im)
{
  void* params[] = {&output, &params0, &re, &im};
  const uintt w = oap::cuda::GetColumns(output);
  const uintt h = oap::cuda::GetRows(output);
  m_cuStatus = execute("CUDAKernel_MultiplyConstant", w, h, params, 0);
}

bool CuProceduresApi::compare (math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, floatt tolerance) {
  if (matrix1 == matrix2) {
    return true;
  }

  if (matrix1 == nullptr && matrix2 != nullptr)
  {
    return false;
  }

  if (matrix2 == nullptr && matrix1 != nullptr)
  {
    return false;
  }

  const uintt w = oap::cuda::GetColumns(matrix1);
  const uintt h = oap::cuda::GetRows(matrix1);

  floatt o = compareProcedure("CUDAKernel_CompareOpt", matrix1, matrix2, w, h, w,
                          h);

  return (-tolerance <= o && o <= tolerance);
}

bool CuProceduresApi::compareVer2(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, floatt tolerance) {
  if (matrix1 == matrix2) {
    return true;
  }

  const uintt w = oap::cuda::GetColumns(matrix1);
  const uintt h = oap::cuda::GetRows(matrix1);

  floatt o = compareProcedure("CUDAKernel_CompareOptVer2", matrix1, matrix2, w, h, w / 2, h);
  return (-tolerance <= o && o <= tolerance);
}

void CuProceduresApi::sigmoid (math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Sigmoid", matrix, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::sigmoid (math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SigmoidDim", matrix, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sigmoid (math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SigmoidDimPeriodic", matrix, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Sigmoid", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::sigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SigmoidDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SigmoidDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DSigmoid", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dsigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DSigmoidDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DSigmoidDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_MultiplyDSigmoid", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::multiplyDSigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_MultiplyDSigmoidDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_MultiplyDSigmoidDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::linear (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (output, matrix);
}

void CuProceduresApi::linear (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  auto minfo = oap::cuda::GetMatrixInfo (output);
  math::MatrixInfo minfo1 (minfo.isRe, minfo.isIm, dim[0], dim[1]);

  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrix (minfo1);

  PRINT_CUMATRIX(dmatrix.get());
  oap::cuda::CopyDeviceToDevice (dmatrix, matrix);
  oap::cuda::SetMatrix (output, dmatrix, 0, 0);
}

void CuProceduresApi::linear (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  debugAssert ("Not implemented yet" == nullptr);
}

void CuProceduresApi::dlinear (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  oap::HostMatrixUPtr hmatrix = oap::host::NewMatrixWithValue (oap::cuda::GetMatrixInfo(output), 1.f);
  oap::cuda::CopyHostMatrixToDeviceMatrix (output, hmatrix);
}

void CuProceduresApi::dlinear (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  auto minfo = oap::cuda::GetMatrixInfo (output);
  math::MatrixInfo minfo1 (minfo.isRe, minfo.isIm, dim[0], dim[1]);

  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrix (minfo1);

  oap::cuda::SetMatrix (output, dmatrix, 0, 0);
}

void CuProceduresApi::dlinear (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  debugAssert ("Not implemented yet" == nullptr);
}

void CuProceduresApi::tanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Tanh", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::tanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_TanhDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::tanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_TanhDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dtanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DTanh", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dtanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DTanhDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dtanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DTanhDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sin (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Sin", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::sin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SinDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SinDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSin (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_MultiplyDSin", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::multiplyDSin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_MultiplyDSinDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_MultiplyDSinDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsin (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DSin", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dsin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DSinDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DSinDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::relu (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Relu", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::relu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_ReluDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::relu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_ReluDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::drelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DRelu", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::drelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DReluDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::drelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DReluDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::prelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_PRelu", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::prelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_PReluDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::prelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_PReluDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dprelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DPRelu", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dprelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DPReluDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dprelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DPReluDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Softplus", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SoftplusDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SoftplusDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsoftplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DSoftplus", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dsoftplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim2 dim)
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DSoftplusDim", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsoftplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix, generic::Dim22 dim)
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DSoftplusDimPeriodic", output, matrix, dim, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::convolve (math::ComplexMatrix* output, const math::ComplexMatrix* matrix, const math::ComplexMatrix* kernel)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(matrix);
  CHECK_MATRIX(kernel);

  m_cuStatus = oap::generic::convolve (output, matrix, kernel, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::poolAverage (math::ComplexMatrix* output, const math::ComplexMatrix* matrix, const math::MatrixDim& kernel)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(matrix);
  logAssert (kernel.columns != 0 && kernel.rows != 0);
  
  m_cuStatus = oap::generic::poolAverage (output, matrix, kernel, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback, m_createKernelArray);
}

floatt CuProceduresApi::mean (const math::ComplexMatrix* matrix)
{
  auto minfo = oap::cuda::GetMatrixInfo (matrix);
  floatt sumF = 0;
  floatt im = 0;
  this->sum (sumF, im, matrix);
  return sumF / (minfo.columns () + minfo.rows ());
}

floatt CuProceduresApi::stddv (const math::ComplexMatrix* matrix, floatt mean)
{
  floatt sd = 0;

  auto minfo = oap::cuda::GetMatrixInfo (matrix);

  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrixFromMatrixInfo (minfo);
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (dmatrix, matrix);

  this->add (dmatrix.get (), dmatrix.get (), -mean);
  this->magnitude (sd, dmatrix.get ());
  sd = sd / ((minfo.columns() * minfo.rows()) - 1);
  return sd;
}

floatt CuProceduresApi::stddv (const math::ComplexMatrix* matrix)
{
  floatt mn = mean (matrix);
  return stddv (matrix, mn);
}

void CuProceduresApi::scale (math::ComplexMatrix* matrix)
{
  floatt mn = this->mean (matrix);
  floatt sd = this->stddv (matrix, mn);
  this->add (matrix, matrix, -mn);
  this->multiplyConstant (matrix, matrix, 1. / sd, 0);
}

void CuProceduresApi::dotProduct (oap::Memory& output, const oap::Memory& arg1, const oap::Memory& arg2, const oap::MemoryRegion_3_Args* regions)
{
  //m_cuStatus = oap::generic::dotProduct (output, arg1, arg2, regions, &m_kernel);
}

floatt CuProceduresApi::compareProcedure(const char* cuKernelName, math::ComplexMatrix* matrix1,
                                math::ComplexMatrix* matrix2, uintt w, uintt h,
                                uintt wthreads, uintt hthreads) {
  if (matrix1 == matrix2) {
    return true;
  }

  uint blocks[2];
  uint threads[2];

  m_kernel.calculateThreadsBlocks(blocks, threads, wthreads, hthreads);

  assert (threads[0] * threads[1] * sizeof(floatt) < oap::cuda::Context::Instance().getSharedMemorySize());

  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);
  m_kernel.setSharedMemory(threads[0] * threads[1] * sizeof(floatt));

  uintt outputLength = blocks[0] * blocks[1];

  m_dcompareOutputBuffer.realloc(outputLength);
  m_hcompareOutputBuffer.realloc(outputLength);

  const void* params[] = {&m_dcompareOutputBuffer.m_buffer, &matrix1, &matrix2};

  m_cuStatus = ::oap::cuda::Kernel::Execute(cuKernelName, params, m_kernel);

  CudaUtils::CopyDeviceToHost(
      m_hcompareOutputBuffer.m_buffer, m_dcompareOutputBuffer.m_buffer,
      outputLength * m_hcompareOutputBuffer.getSizeOfType());

  floatt outcome = 0;
  for (uint fa = 0; fa < blocks[0] * blocks[1]; ++fa) {
    outcome += m_hcompareOutputBuffer.m_buffer[fa];
  }

  outcome = outcome / (w * h);

  m_compareOperationOutput = outcome;
  return outcome;
}

floatt CuProceduresApi::magnitude2Procedure(const char* cuKernelName,
                                     math::ComplexMatrix* matrix, uintt wthreads,
                                     uintt hthreads)
{
  uint blocks[2];
  uint threads[2];

  m_kernel.calculateThreadsBlocks(blocks, threads, wthreads, hthreads);

  assert (threads[0] * threads[1] * sizeof(floatt) < oap::cuda::Context::Instance().getSharedMemorySize());

  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);
  m_kernel.setSharedMemory(threads[0] * threads[1] * sizeof(floatt));

  uintt outputLength = blocks[0] * blocks[1];

  m_dmagnitudeOutputBuffer.realloc(outputLength);
  m_hmagnitudeOutputBuffer.realloc(outputLength);

  const void* params[] = {&m_dmagnitudeOutputBuffer.m_buffer, &matrix};

  m_cuStatus = ::oap::cuda::Kernel::Execute(cuKernelName, params, m_kernel);

  return magnitude2Procedure_GetOutput(blocks, outputLength);
}

floatt CuProceduresApi::magnitude2Procedure_GetOutput(uint blocks[2], uintt outputLength) const {

  const uintt size = outputLength * m_dmagnitudeOutputBuffer.getSizeOfType();
  CudaUtils::CopyDeviceToHost( m_hmagnitudeOutputBuffer.m_buffer, m_dmagnitudeOutputBuffer.m_buffer, size);

  floatt outcome = 0;
  for (uint fa = 0; fa < blocks[0] * blocks[1]; ++fa) {
    outcome += m_hmagnitudeOutputBuffer.m_buffer[fa];
  }

  return outcome;
}

void CuProceduresApi::qrProcedure(QRType qrType, math::ComplexMatrix* Q, math::ComplexMatrix* R,
                           math::ComplexMatrix* A, math::ComplexMatrix* AT, math::ComplexMatrix* P,
                           math::ComplexMatrix* I, math::ComplexMatrix* v, math::ComplexMatrix* vt,
                           math::ComplexMatrix* vvt) {
  uint blocks[2];
  uint threads[2];
  const uintt w = oap::cuda::GetColumns(A);
  const uintt h = oap::cuda::GetRows(A);

  m_kernel.calculateThreadsBlocks(blocks, threads, w, h);
  m_kernel.setBlocksCount(blocks[0], blocks[1]);
  m_kernel.setThreadsCount(threads[0], threads[1]);

  m_dqrSums.realloc(blocks[0] * blocks[1]);

  if (qrType == OPT) {
    m_kernel.setSharedMemory(h * sizeof(floatt));
    const void* params[] = {&Q, &R, &A, &AT, &m_dqrSums.m_buffer, &P, &I, &v, &vt,
                      &vvt};
    m_cuStatus =
        ::oap::cuda::Kernel::Execute("CUDAKernel_QRHTOpt", params, m_kernel);
  } else {
    m_dqrBuffer.realloc(h);
    const void* params[] = {&Q, &R, &A, &AT, &m_dqrSums.m_buffer,
                      &m_dqrBuffer.m_buffer, &P, &I, &v, &vt, &vvt};
    m_cuStatus = ::oap::cuda::Kernel::Execute("CUDAKernel_QRHT", params, m_kernel);
  }
}

floatt CuProceduresApi::getCompareOperationSum() const {
  return m_compareOperationOutput;
}

std::string CuProceduresApi::getMsgStatus() const { return m_kernel.getErrorMsg(); }

void CuProceduresApi::crossEntropy(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  oap::generic::crossEntropy (output, params0, params1, &m_kernel, m_bmApi);
}

uintt CuProceduresApi::GetColumns (const math::ComplexMatrix* matrix)
{
  return oap::cuda::GetColumns (matrix);
}

uintt CuProceduresApi::GetRows (const math::ComplexMatrix* matrix)
{
  return oap::cuda::GetRows (matrix);
}

void CuProceduresApi::deallocKernelArrays ()
{
  for (auto it = m_kernelArrays.begin(); it != m_kernelArrays.end(); ++it)
  {
    CudaUtils::FreeDeviceMem (it->second);
  }
  m_kernelArrays.clear();
}

void CuProceduresApi::addDotProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
#ifdef DEBUG
  CHECK_MATRIX(outputs);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);
#endif
  const uintt output_columns = oap::cuda::GetColumns(outputs);
  const uintt output_rows = oap::cuda::GetRows(outputs);

  addDotProduct(outputs, params0, params1, output_columns, output_rows);
}

void CuProceduresApi::tensorProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
#ifdef DEBUG
  CHECK_MATRIX(outputs);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);
#endif

  const uintt output_columns = oap::cuda::GetColumns(outputs);
  const uintt output_rows = oap::cuda::GetRows(outputs);

  tensorProduct (outputs, params0, params1, output_columns, output_rows);
}

void CuProceduresApi::elementWiseProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  m_cuStatus = oap::generic::hadamardProduct (outputs, params0, params1, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::schurProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  hadamardProduct (outputs, params0, params1);
}

void CuProceduresApi::tensorProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt outputD[2], uintt matrix1D[2], uintt matrix2D[2])
{
  generic::Dim32 dim {{{outputD[0], outputD[1]}, {matrix1D[0], matrix1D[1]}, {matrix2D[0], matrix2D[1]}}};
  tensorProduct (outputs, params0, params1, dim);
}

void CuProceduresApi::dotProductOpt(math::ComplexMatrix* outputs, math::ComplexMatrix* params0,
                                    math::ComplexMatrix* params1) {
  const uintt ocolumns = oap::cuda::GetColumns(outputs);
  const uintt orows = oap::cuda::GetRows(outputs);
  const uintt p1rows = oap::cuda::GetRows(params0);
  const uintt p2columns = oap::cuda::GetColumns(params1);
  dotProductOpt(outputs, params0, params1, ocolumns, orows, p1rows, p2columns);
}

void CuProceduresApi::subtract(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  const uintt columns = oap::cuda::GetColumns(outputs);
  const uintt rows = oap::cuda::GetRows(outputs);
  subtract(outputs, params0, params1, columns, rows);
}

void CuProceduresApi::addSubstract(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  const uintt columns = oap::cuda::GetColumns(outputs);
  const uintt rows = oap::cuda::GetRows(outputs);
  addSubstract(outputs, params0, params1, columns, rows);
}

void CuProceduresApi::add (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  const uintt columns = oap::cuda::GetColumns(outputs);
  const uintt rows = oap::cuda::GetRows(outputs);
  add(outputs, params0, params1, columns, rows);
}

}
