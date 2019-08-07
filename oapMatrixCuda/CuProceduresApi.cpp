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
#include "oapHostMatrixUPtr.h"
#include "oapHostMatrixPtr.h"

#include "ThreadsMapper.h"
#include "oapCudaMatrixUtils.h"

#include "GenericProceduresApi.h"

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
      m_preExecCallback (std::bind(&CuProceduresApi::resetFlags, this))
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

void CuProceduresApi::dotProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::dotProduct (output, params0, params1, columns, rows, &m_kernel, m_preExecCallback, m_bmApi);
}

void CuProceduresApi::addDotProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows)
{
#ifdef CU_PROCEDURES_API_PRINT
  debug(__func__);
#endif
  CHECK_MATRIX(output);
  CHECK_MATRIX(params0);
  CHECK_MATRIX(params1);

  oap::generic::check_dotProduct (output, params0, params1, columns, rows, m_bmApi);

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

  oap::generic::BasicMatrixDimApi<decltype(CuProceduresApi::GetColumns), decltype(CuProceduresApi::GetRows)> bmdApi (CuProceduresApi::GetColumns, CuProceduresApi::GetRows);
  oap::generic::check_tensorProduct (output, params0, params1, columns, rows, bmdApi);

  void* params[] = {&output, &params0, &params1};
  const char* kname = "CUDAKernel_TensorProduct";

  m_cuStatus = generic::executeKernel (kname, output, params, &m_kernel, m_bmApi, m_preExecCallback);
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

void CuProceduresApi::dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                                 uintt outputD[2], uintt matrix1D[2], uintt matrix2D[2])
{
  oap::generic::dotProduct (output, matrix1, matrix2, outputD, matrix1D, matrix2D, &m_kernel, m_preExecCallback, m_bmApi,
                            std::bind(&CuProceduresApi::createKernelArray, this, std::placeholders::_1, std::placeholders::_2));
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
  const uintt p1rows = CudaUtils::GetRows(params0);
  const uintt p2columns = CudaUtils::GetColumns(params1);
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

void CuProceduresApi::setVector(math::Matrix* V, uintt column, math::Matrix* v,
                         uintt length) {
  const uintt w = CudaUtils::GetColumns(v);
  const uintt h = CudaUtils::GetRows(v);
  void* params[] = {&V, &column, &v, &length};
  m_cuStatus = execute("CUDAKernel_SetVector", w, h, params, 0);
}

void CuProceduresApi::getVector(math::Matrix* vector, uintt length,
                         math::Matrix* matrix, uintt column) {
  const uintt w = CudaUtils::GetColumns(vector);
  const uintt h = CudaUtils::GetRows(vector);
  void* params[] = {&vector, &length, &matrix, &column};
  m_cuStatus = execute("CUDAKernel_GetVector", w, h, params, 0);
}

void CuProceduresApi::getVector(math::Matrix* vector, math::Matrix* matrix,
                         uintt column) {
  const uintt w = CudaUtils::GetColumns(vector);
  const uintt h = CudaUtils::GetRows(vector);
  uintt length = w * h;
  void* params[] = {&vector, &length, &matrix, &column};
  m_cuStatus = execute("CUDAKernel_GetVector", w, h, params, 0);
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
  const uintt w = CudaUtils::GetColumns(params0);
  const uintt h = CudaUtils::GetRows(params0);
  output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
}

void CuProceduresApi::magnitude2OptVer2(floatt& output, math::Matrix* params0) {
  const uintt w = CudaUtils::GetColumns(params0);
  const uintt h = CudaUtils::GetRows(params0);
  if (w > 1) {
    output =
        magnitude2Procedure("CUDAKernel_MagnitudeOptVer2", params0, w / 2, h);
  } else {
    output = magnitude2Procedure("CUDAKernel_MagnitudeOpt", params0, w, h);
  }
}

void CuProceduresApi::setDiagonal(math::Matrix* matrix, floatt re, floatt im) {
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);
  void* params[] = {&matrix, &re, &im};
  m_cuStatus = execute("CUDAKernel_SetDiagonal", w, h, params, 0);
}

void CuProceduresApi::setIdentity(math::Matrix* matrix) {
  void* params[] = {&matrix};
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);
  m_cuStatus = execute("CUDAKernel_SetIdentity", w, h, params, 0);
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
  const uintt w = CudaUtils::GetColumns(H);
  const uintt h = CudaUtils::GetRows(H);
  if (maxThreads >= w * h) {
    m_cuStatus = execute("CUDAKernel_QRGR", w, h, params, 0);
  } else {
    m_cuStatus = HOSTKernel_QRGR(Q, R, H, aux0, aux1, aux2, aux3, m_kernel);
  }
}

bool CuProceduresApi::isUpperTriangular(math::Matrix* matrix) {
  int result = -10;
  void* params[] = {&m_doutputIsTriangular, &matrix};
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);
  m_cuStatus = execute("CUDAKernel_IsUpperTriangular", w, h, params, 0);
  CudaUtils::CopyDeviceToHost(&result, m_doutputIsTriangular, sizeof(int));
  return result == 1;
}

void CuProceduresApi::calcTriangularHStep(math::Matrix* H, math::Matrix* Q, math::Matrix* R,
                                   math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
                                   math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  uint w = CudaUtils::GetColumns(H);
  uint h = CudaUtils::GetRows(H);
  calcTriangularHStep(H, Q, R,
                      aux1, aux2, aux3,
                      aux4, aux5, aux6,
                      w, h);

}

void CuProceduresApi::calcTriangularHStep(math::Matrix* H, math::Matrix* Q, math::Matrix* R,
                                   math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
                                   math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6,
                                   uint columns, uint rows)
{
  void* params[] = {&H, &Q, &R, &aux1, &aux2, &aux3, &aux4, &aux5, &aux6};
  const uintt w = columns;
  const uintt h = rows;
  m_cuStatus = execute("CUDAKernel_CalculateTriangularHStep", w, h, params, 0);
}

void CuProceduresApi::multiplyReConstant(math::Matrix* output, math::Matrix* params0, floatt re)
{
  void* params[] = {&output, &params0, &re};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
  m_cuStatus = execute("CUDAKernel_MultiplyConstantRe", w, h, params, 0);
}

void CuProceduresApi::multiplyConstant(math::Matrix* output, math::Matrix* params0,
                                             floatt re, floatt im)
{
  void* params[] = {&output, &params0, &re, &im};
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);
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

  const uintt w = CudaUtils::GetColumns(matrix1);
  const uintt h = CudaUtils::GetRows(matrix1);

  floatt o = compareProcedure("CUDAKernel_CompareOpt", matrix1, matrix2, w, h, w,
                          h);

  return (-tolerance <= o && o <= tolerance);
}

bool CuProceduresApi::compareVer2(math::Matrix* matrix1, math::Matrix* matrix2, floatt tolerance) {
  if (matrix1 == matrix2) {
    return true;
  }

  const uintt w = CudaUtils::GetColumns(matrix1);
  const uintt h = CudaUtils::GetRows(matrix1);

  floatt o = compareProcedure("CUDAKernel_CompareOptVer2", matrix1, matrix2, w, h,
                          w / 2, h);
  return (-tolerance <= o && o <= tolerance);
}

void CuProceduresApi::sigmoid (math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns(matrix);
  const uintt h = CudaUtils::GetRows(matrix);

  void* params[] = {&matrix, &matrix};

  m_cuStatus = execute("CUDAKernel_Sigmoid", w, h, params, 0);
}

void CuProceduresApi::sigmoid (math::Matrix* output, math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns (output);
  const uintt h = CudaUtils::GetRows (output);

  void* params[] = {&output, &matrix};

  m_cuStatus = execute("CUDAKernel_Sigmoid", w, h, params, 0);
}

void CuProceduresApi::sigmoidDerivative (math::Matrix* omatrix, math::Matrix* imatrix)
{
  const uintt w = CudaUtils::GetColumns(omatrix);
  const uintt h = CudaUtils::GetRows(omatrix);

  void* params[] = {&omatrix, &imatrix};

  m_cuStatus = execute("CUDAKernel_SigmoidDerivative", w, h, params, 0);
}

void CuProceduresApi::multiplySigmoidDerivative(math::Matrix* omatrix, math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns(omatrix);
  const uintt h = CudaUtils::GetRows(omatrix);

  void* params[] = {&omatrix, &matrix};

  m_cuStatus = execute("CUDAKernel_MultiplySigmoidDerivative", w, h, params, 0);
}

void CuProceduresApi::linear (math::Matrix* output, math::Matrix* matrix)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (output, matrix);
}

void CuProceduresApi::linearDerivative (math::Matrix* output, math::Matrix* matrix)
{
  oap::HostMatrixUPtr hmatrix = oap::host::NewMatrix (oap::cuda::GetMatrixInfo(output), 1.f);
  oap::cuda::CopyHostMatrixToDeviceMatrix (output, hmatrix);
}

void CuProceduresApi::tanh (math::Matrix* output, const math::Matrix* matrix)
{
  m_cuStatus = oap::generic::executeKernel1Arg ("CUDAKernel_Tanh", output, matrix, &m_kernel, m_bmApi, true,
               m_preExecCallback);
}

void CuProceduresApi::tanhDerivative (math::Matrix* output, const math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);

  void* params[] = {&output, &matrix};

  m_cuStatus = execute("CUDAKernel_TanhDerivative", w, h, params, 0);

}

void CuProceduresApi::sin (math::Matrix* output, const math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);

  void* params[] = {&output, &matrix};

  m_cuStatus = execute("CUDAKernel_Sin", w, h, params, 0);
}

void CuProceduresApi::multiplySinDerivative (math::Matrix* output, math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);

  void* params[] = {&output, &matrix};

  m_cuStatus = execute("CUDAKernel_MultiplySinDerivative", w, h, params, 0);
}

void CuProceduresApi::sinDerivative (math::Matrix* output, const math::Matrix* matrix)
{
  const uintt w = CudaUtils::GetColumns(output);
  const uintt h = CudaUtils::GetRows(output);

  void* params[] = {&output, &matrix};

  m_cuStatus = execute("CUDAKernel_SinDerivative", w, h, params, 0);
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
  return CudaUtils::GetColumns (matrix);
}

uintt CuProceduresApi::GetRows (const math::Matrix* matrix)
{
  return CudaUtils::GetRows (matrix);
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
