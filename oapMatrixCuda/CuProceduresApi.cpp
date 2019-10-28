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

#include "GenericProceduresApi.h"
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

  return ::oap::cuda::Kernel::Execute(functionName, params, m_kernel);
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

void CuProceduresApi::dotProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
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

void CuProceduresApi::dotProductShared (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::dotProductShared (output, params0, params1, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::addDotProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::check_dotProduct (output, params0, params1, m_bmApi);

  void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_AddDotProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::tensorProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::check_tensorProduct (output, params0, params1, columns, rows, m_bmApi);

  void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_TensorProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::tensorProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, uintt dims[3][2])
{
  oap::generic::tensorProduct (output, matrix1, matrix2, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::hadamardProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::BasicMatrixDimApi<decltype(CuProceduresApi::GetColumns), decltype(CuProceduresApi::GetRows)> bmdApi (CuProceduresApi::GetColumns, CuProceduresApi::GetRows);
  oap::generic::check_hadamardProduct (output, params0, params1, columns, rows, bmdApi);

  void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_HadamardProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::hadamardProductVec (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::BasicMatrixDimApi<decltype(CuProceduresApi::GetColumns), decltype(CuProceduresApi::GetRows)> bmdApi (CuProceduresApi::GetColumns, CuProceduresApi::GetRows);
  oap::generic::check_hadamardProductVec (output, params0, params1, columns, rows, bmdApi);

  void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_PHadamardProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::calculateQTHQ(math::Matrix* output, math::Matrix* H,
                             math::Matrix* Q, math::Matrix* aux) {
  transpose(output, Q);
  dotProduct(aux, H, output);
  dotProduct(output, Q, aux);
}

void CuProceduresApi::dotProductEx(math::Matrix* output, math::Matrix* params0,
                                  math::Matrix* params1, MatrixEx* matrixEx,
                                  uintt columns, uintt rows)
{
  void* params[] = {&output, &params0, &params1, &matrixEx};
  const char* kname = "CUDAKernel_DotProductEx";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dotProductPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2)
{
  m_cuStatus = oap::generic::dotProductPeriodic (output, matrix1, matrix2, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, uintt dims[3][2], uintt periodicRows)
{
  m_cuStatus = oap::generic::dotProductDimPeriodic (output, matrix1, matrix2, dims, periodicRows, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                                 uintt dims[3][2])
{
  m_cuStatus = oap::generic::dotProduct (output, matrix1, matrix2, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dotProductOpt(math::Matrix* output, math::Matrix* params0,
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
  m_cuStatus =
      execute("CUDAKernel_DotProductOpt", ocolumns, orows, params, size);
}

void CuProceduresApi::dotProductExOpt(math::Matrix* output, math::Matrix* params0,
                               math::Matrix* params1, MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &params1, &matrixEx};
  bool isRe = CudaUtils::GetReValues(output) != NULL;
  bool isIm = CudaUtils::GetImValues(output) != NULL;
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

void CuProceduresApi::transposeEx(math::Matrix* output, math::Matrix* params0,
                                 MatrixEx* matrixEx) {
  void* params[] = {&output, &params0, &matrixEx};
  const uintt w = CudaUtils::GetColumns(matrixEx);
  const uintt h = CudaUtils::GetRows(matrixEx);
  execute("CUDAKernel_TransposeEx", w, h, params, 0);
}

void CuProceduresApi::transpose(math::Matrix* output, math::Matrix* params0) {
  const uintt wo = oap::cuda::GetColumns (output);
  const uintt ho = oap::cuda::GetRows (output);

  const uintt wp = oap::cuda::GetColumns (params0);
  const uintt hp = oap::cuda::GetRows (params0);

  debugAssert (ho == wp && hp == wo);

  if ((wo == 1 && ho == wp && hp == 1) || (ho == 1 && hp == wo && wp == 1))
  {
    oap::cuda::CopyDeviceToDevice(output, params0);
  }
  else
  {
    void* params[] = {&output, &params0};
    m_cuStatus = execute("CUDAKernel_Transpose", wo, ho, params, 0);
  }
}

void CuProceduresApi::conjugateTranspose(math::Matrix* output, math::Matrix* params0) {

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

void CuProceduresApi::substract(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
  void* params[] = {&output, &params0, &params1};
  m_cuStatus = execute("CUDAKernel_Substract", columns, rows, params, 0);
}

void CuProceduresApi::addSubstract(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
  void* params[] = {&output, &params0, &params1};
  m_cuStatus = execute("CUDAKernel_AddSubstract", columns, rows, params, 0);
}

void CuProceduresApi::add(math::Matrix* output, math::Matrix* params0,
                   math::Matrix* params1, uintt columns, uintt rows) {
  void* params[] = {&output, &params0, &params1};
  m_cuStatus = execute("CUDAKernel_Add", columns, rows, params, 0);
}

void CuProceduresApi::setVector (math::Matrix* V, uintt column, math::Matrix* v, uintt length)
{
  m_cuStatus = oap::generic::setVector (V, column, v, length, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::getVector (math::Matrix* vector, uintt length, math::Matrix* matrix, uintt column)
{
  m_cuStatus = oap::generic::getVector (vector, length, matrix, column, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::getVector (math::Matrix* vector, math::Matrix* matrix, uintt column)
{
  m_cuStatus = oap::generic::getVector (vector, matrix, column, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::magnitude(floatt& output, math::Matrix* param0) {
  magnitude2(output, param0);
  output = sqrt(output);
}
/*
void CuProceduresApi::sum (floatt& output, math::Matrix* matrix)
{
  const uintt w = oap::cuda::GetColumns (matrix);
  const uintt h = oap::cuda::GetRows (matrix);
  void* params[] = {&output, &matrix};

  prepareDims (w, h);

  output = execute ("CUDAKernel_Sum", w, h, params, 0, false);
}
*/
void CuProceduresApi::sum (floatt& reoutput, floatt& imoutput, math::Matrix* matrix)
{
  using HBuffer = oap::TBuffer<floatt, oap::Type::HOST>;
  using DBuffer = oap::TBuffer<floatt, oap::Type::CUDA>;

  using GetAddressType = std::function<floatt*(const math::Matrix*)>;
  using GetAddressTypeRef = GetAddressType&;

  GetAddressType getReValues = [](const math::Matrix* matrix) -> floatt*
  {
    return CudaUtils::GetReValues (matrix);
  };

  GetAddressType getImValues = [](const math::Matrix* matrix) -> floatt*
  {
    return CudaUtils::GetImValues (matrix);
  };

  generic::SumApi<decltype(oap::cuda::GetMatrixInfo), decltype(CudaUtils::CopyDeviceToHost), GetAddressTypeRef>
  sumApi (oap::cuda::GetMatrixInfo, CudaUtils::CopyDeviceToHost, getReValues, getImValues);

  generic::SumBuffers<HBuffer, DBuffer>
  sumBuffers (m_hsumsReBuffer, m_dsumsReBuffer, m_hsumsImBuffer, m_dsumsImBuffer);

  generic::sum (reoutput, imoutput, matrix, &m_kernel, sumApi, sumBuffers);
}

void CuProceduresApi::sum (floatt& reoutput, math::Matrix* matrix)
{
  floatt imoutput;
  sum (reoutput, imoutput, matrix);
}

void CuProceduresApi::magnitudeOpt(floatt& output, math::Matrix* param0) {
  magnitude2Opt(output, param0);
  output = sqrt(output);
}

void CuProceduresApi::magnitudeOptVer2(floatt& output, math::Matrix* param0) {
  magnitude2OptVer2(output, param0);
  output = sqrt(output);
}

void CuProceduresApi::magnitude2(floatt& output, math::Matrix* param0) {
  magnitude2Opt(output, param0);
}

void CuProceduresApi::magnitude2Opt(floatt& output, math::Matrix* params0) {
  const uintt w = oap::cuda::GetColumns(params0);
  const uintt h = oap::cuda::GetRows(params0);
  output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
}

void CuProceduresApi::magnitude2OptVer2(floatt& output, math::Matrix* params0) {
  const uintt w = oap::cuda::GetColumns(params0);
  const uintt h = oap::cuda::GetRows(params0);
  if (w > 1) {
    output =
        magnitude2Procedure("CUDAKernel_MagnitudeOptVer2", params0, w / 2, h);
  } else {
    output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
  }
}

void CuProceduresApi::setDiagonal(math::Matrix* matrix, floatt re, floatt im) {
  const uintt w = oap::cuda::GetColumns(matrix);
  const uintt h = oap::cuda::GetRows(matrix);
  void* params[] = {&matrix, &re, &im};
  m_cuStatus = execute("CUDAKernel_SetDiagonal", w, h, params, 0);
}

void CuProceduresApi::setIdentity (math::Matrix* matrix)
{
  m_cuStatus = oap::generic::setIdentityMatrix (matrix, &m_kernel, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

void CuProceduresApi::setZeroMatrix(math::Matrix* matrix) {
  CudaUtils::SetZeroMatrix(matrix, true, true);
  m_cuStatus = CUDA_SUCCESS;
}

void CuProceduresApi::QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* H,
                    math::Matrix* aux0, math::Matrix* aux1, math::Matrix* aux2,
                    math::Matrix* aux3) {
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

void CuProceduresApi::QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT)
{
  m_cuStatus = oap::generic::qrDecomposition_HT (Q, R, A, V, VT, P, VVT, &m_kernel, *this, oap::cuda::GetMatrixInfo, m_preExecCallback);
}

bool CuProceduresApi::isUpperTriangular(math::Matrix* matrix) {
  int result = -10;
  void* params[] = {&m_doutputIsTriangular, &matrix};
  const uintt w = oap::cuda::GetColumns(matrix);
  const uintt h = oap::cuda::GetRows(matrix);
  m_cuStatus = execute("CUDAKernel_IsUpperTriangular", w, h, params, 0);
  CudaUtils::CopyDeviceToHost(&result, m_doutputIsTriangular, sizeof(int));
  return result == 1;
}

void CuProceduresApi::calcTriangularH (math::Matrix* H, math::Matrix* Q, math::Matrix* R,
                                       math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
                                       math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  void* params[] = {&H, &Q, &R, &aux1, &aux2, &aux3, &aux4, &aux5, &aux6};
  const uintt w = oap::cuda::GetColumns (H);
  const uintt h = oap::cuda::GetRows (H);
  m_cuStatus = execute("CUDAKernel_CalculateTriangularH", w, h, params, 0);
}

void CuProceduresApi::calcTriangularHStep (math::Matrix* H, math::Matrix* Q, math::Matrix* R,
                                           math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
                                           math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  void* params[] = {&H, &Q, &R, &aux1, &aux2, &aux3, &aux4, &aux5, &aux6};
  const uintt w = oap::cuda::GetColumns (H);
  const uintt h = oap::cuda::GetRows (H);
  m_cuStatus = execute("CUDAKernel_CalculateTriangularHStep", w, h, params, 0);
}

void CuProceduresApi::multiplyReConstant(math::Matrix* output, math::Matrix* params0, floatt re)
{
  void* params[] = {&output, &params0, &re};
  const uintt w = oap::cuda::GetColumns(output);
  const uintt h = oap::cuda::GetRows(output);
  m_cuStatus = execute("CUDAKernel_MultiplyConstantRe", w, h, params, 0);
}

void CuProceduresApi::multiplyConstant(math::Matrix* output, math::Matrix* params0,
                                             floatt re, floatt im)
{
  void* params[] = {&output, &params0, &re, &im};
  const uintt w = oap::cuda::GetColumns(output);
  const uintt h = oap::cuda::GetRows(output);
  m_cuStatus = execute("CUDAKernel_MultiplyConstant", w, h, params, 0);
}

bool CuProceduresApi::compare (math::Matrix* matrix1, math::Matrix* matrix2, floatt tolerance) {
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

bool CuProceduresApi::compareVer2(math::Matrix* matrix1, math::Matrix* matrix2, floatt tolerance) {
  if (matrix1 == matrix2) {
    return true;
  }

  const uintt w = oap::cuda::GetColumns(matrix1);
  const uintt h = oap::cuda::GetRows(matrix1);

  floatt o = compareProcedure("CUDAKernel_CompareOptVer2", matrix1, matrix2, w, h,
                          w / 2, h);
  return (-tolerance <= o && o <= tolerance);
}

void CuProceduresApi::sigmoid (math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Sigmoid", matrix, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::sigmoid (math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SigmoidDim", matrix, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sigmoid (math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SigmoidDimPeriodic", matrix, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sigmoid (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Sigmoid", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::sigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SigmoidDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SigmoidDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsigmoid (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DSigmoid", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dsigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DSigmoidDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DSigmoidDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSigmoid (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_MultiplyDSigmoid", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::multiplyDSigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_MultiplyDSigmoidDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSigmoid (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_MultiplyDSigmoidDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::linear (math::Matrix* output, math::Matrix* matrix)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (output, matrix);
}

void CuProceduresApi::linear (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  auto minfo = oap::cuda::GetMatrixInfo (output);
  math::MatrixInfo minfo1 (minfo.isRe, minfo.isIm, dims[0], dims[1]);

  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrix (minfo1, 1.f);

  oap::cuda::CopyDeviceToDevice (dmatrix, matrix);
  oap::cuda::SetMatrix (output, dmatrix, 0, 0);
}

void CuProceduresApi::linear (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  debugAssert ("Not implemented yet" == nullptr);
}

void CuProceduresApi::dlinear (math::Matrix* output, math::Matrix* matrix)
{
  oap::HostMatrixUPtr hmatrix = oap::host::NewMatrix (oap::cuda::GetMatrixInfo(output), 1.f);
  oap::cuda::CopyHostMatrixToDeviceMatrix (output, hmatrix);
}

void CuProceduresApi::dlinear (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  auto minfo = oap::cuda::GetMatrixInfo (output);
  math::MatrixInfo minfo1 (minfo.isRe, minfo.isIm, dims[0], dims[1]);

  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrix (minfo1, 1.f);

  oap::cuda::SetMatrix (output, dmatrix, 0, 0);
}

void CuProceduresApi::dlinear (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  debugAssert ("Not implemented yet" == nullptr);
}

void CuProceduresApi::tanh (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Tanh", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::tanh (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_TanhDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::tanh (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_TanhDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dtanh (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DTanh", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dtanh (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DTanhDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dtanh (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DTanhDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sin (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_Sin", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::sin (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_SinDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::sin (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_SinDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSin (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_MultiplyDSin", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::multiplyDSin (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_MultiplyDSinDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::multiplyDSin (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_MultiplyDSinDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsin (math::Matrix* output, math::Matrix* matrix)
{
  m_cuStatus = oap::generic::func ("CUDAKernel_DSin", output, matrix, &m_kernel, m_bmApi, m_preExecCallback);
}

void CuProceduresApi::dsin (math::Matrix* output, math::Matrix* matrix, uintt dims[2])
{
  m_cuStatus = oap::generic::funcDim ("CUDAKernel_DSinDim", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

void CuProceduresApi::dsin (math::Matrix* output, math::Matrix* matrix, uintt dims[2][2])
{
  m_cuStatus = oap::generic::funcDimPeriodic ("CUDAKernel_DSinDimPeriodic", output, matrix, dims, &m_kernel, m_bmApi, m_preExecCallback, m_createKernelArray);
}

floatt CuProceduresApi::compareProcedure(const char* cuKernelName, math::Matrix* matrix1,
                                math::Matrix* matrix2, uintt w, uintt h,
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

  void* params[] = {&m_dcompareOutputBuffer.m_buffer, &matrix1, &matrix2};

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
                                     math::Matrix* matrix, uintt wthreads,
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

  void* params[] = {&m_dmagnitudeOutputBuffer.m_buffer, &matrix};

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

void CuProceduresApi::qrProcedure(QRType qrType, math::Matrix* Q, math::Matrix* R,
                           math::Matrix* A, math::Matrix* AT, math::Matrix* P,
                           math::Matrix* I, math::Matrix* v, math::Matrix* vt,
                           math::Matrix* vvt) {
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
    void* params[] = {&Q, &R, &A, &AT, &m_dqrSums.m_buffer, &P, &I, &v, &vt,
                      &vvt};
    m_cuStatus =
        ::oap::cuda::Kernel::Execute("CUDAKernel_QRHTOpt", params, m_kernel);
  } else {
    m_dqrBuffer.realloc(h);
    void* params[] = {&Q, &R, &A, &AT, &m_dqrSums.m_buffer,
                      &m_dqrBuffer.m_buffer, &P, &I, &v, &vt, &vvt};
    m_cuStatus = ::oap::cuda::Kernel::Execute("CUDAKernel_QRHT", params, m_kernel);
  }
}

floatt CuProceduresApi::getCompareOperationSum() const {
  return m_compareOperationOutput;
}

std::string CuProceduresApi::getMsgStatus() const { return m_kernel.getErrorMsg(); }

void CuProceduresApi::crossEntropy(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  oap::generic::crossEntropy (output, params0, params1, &m_kernel, m_bmApi);
}

uintt CuProceduresApi::GetColumns (const math::Matrix* matrix)
{
  return oap::cuda::GetColumns (matrix);
}

uintt CuProceduresApi::GetRows (const math::Matrix* matrix)
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

}
