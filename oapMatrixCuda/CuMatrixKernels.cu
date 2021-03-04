#include "CuMatrixProcedures.h"
#include "CuMatrixUtils.h"

extern "C" __global__ void CUDAKernel_AddDotProductRe(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  CUDA_addDotProductRe(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddDotProductIm(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  CUDA_addDotProductIm(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddDotProduct(math::ComplexMatrix* output,
                                                 math::ComplexMatrix* params0,
                                                 math::ComplexMatrix* params1) {
  CUDA_addDotProduct(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProductRe(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  CUDA_specific_dotProductRe (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProductIm(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  CUDA_specific_dotProductIm(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProduct(math::ComplexMatrix* output,
                                                 math::ComplexMatrix* params0,
                                                 math::ComplexMatrix* params1) {
  CUDA_specific_dotProduct (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProductShared (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDAKernel_dotProductShared (output, params0, params1);
}

extern "C" __global__ void
CUDAKernel_DotProductPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_dotProductPeriodic (output, params0, params1);
}

extern "C" __global__ void
CUDAKernel_DotProductDimPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt* ex)
{
  CUDA_dotProductDimPeriodic (output, params0, params1, ex);
}

extern "C" __global__
void CUDAKernel_DotProductDim(
     math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt* ex)
{
  CUDA_dotProductDim (output, params0, params1, ex);
}

extern "C" __global__ void CUDAKernel_AddReMatrices (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addReMatrices (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddImMatrices (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addImMatrices (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddMatrices (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addMatrices (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddReMatrixValue (math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt params1)
{
  CUDA_addReMatrixValue (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddImMatrixValue (math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt params1)
{
  CUDA_addImMatrixValue (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddMatrixValue (math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt params1)
{
  CUDA_addMatrixValue (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SubstractRe(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_subtractReMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SubstractIm(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_subtractImMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SubstractReal(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_subtractRealMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddSubstractRe(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addSubstractReMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddSubstractIm(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addSubstractImMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddSubstractReal(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addSubstractRealMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_Substract(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_subtractMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddSubstract(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_addSubstractMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantRe(math::ComplexMatrix* output,
                                                         math::ComplexMatrix* params0,
                                                         floatt re) {
  CUDA_multiplyConstantReMatrix(output, params0, re);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantIm(math::ComplexMatrix* output,
                                                         math::ComplexMatrix* params0,
                                                         floatt im) {
  CUDA_multiplyConstantImMatrix(output, params0, im);
}

extern "C" __global__ void CUDAKernel_MultiplyConstant(math::ComplexMatrix* output,
                                                       math::ComplexMatrix* params0,
                                                       floatt re, floatt im) {
  CUDA_multiplyConstantMatrix(output, params0, re, im);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantReal(
    math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt re, floatt im) {
  CUDA_multiplyConstantRealMatrix(output, params0, re, im);
}

extern "C" __global__ void CUDAKernel_TransposeRe(math::ComplexMatrix* output,
                                                  math::ComplexMatrix* params0) {
  CUDA_transposeReMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_TransposeIm(math::ComplexMatrix* output,
                                                  math::ComplexMatrix* params0) {
  CUDA_transposeImMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_Transpose(math::ComplexMatrix* output,
                                                math::ComplexMatrix* params0) {
  CUDA_transposeMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_TransposeEx(math::ComplexMatrix* output,
                                                  math::ComplexMatrix* params0,
                                                  MatrixEx* matrixEx) {
  CUDA_transposeMatrixEx(output, params0, *matrixEx);
}

extern "C" __global__ void CUDAKernel_ConjugateTranspose(math::ComplexMatrix* output,
                                                math::ComplexMatrix* params0) {
  CUDA_conjugateTransposeMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_Magnitude(floatt* value,
                                                math::ComplexMatrix* params0,
                                                floatt* buffer) {
  CUDA_magnitudeOpt(value, params0, buffer);
}

extern "C" __global__ void CUDAKernel_QRGR(
    math::ComplexMatrix* output0, math::ComplexMatrix* output1, math::ComplexMatrix* params0,
    math::ComplexMatrix* aux0, math::ComplexMatrix* aux1, math::ComplexMatrix* aux2,
    math::ComplexMatrix* aux3) {
  CUDA_QRGR(output0, output1, params0, aux0, aux1, aux2, aux3);
}

extern "C" __global__ void CUDAKernel_QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, math::ComplexMatrix* P, math::ComplexMatrix* VVT)
{
  CudaKernel_QRHT (Q, R, A, V, VT, P, VVT);
}

extern "C" __global__ void CUDAKernel_SetVector (math::ComplexMatrix* output, uintt index, math::ComplexMatrix* params0, uintt length)
{
  CUDAKernel_setVector(output, index, params0, length);
}

extern "C" __global__ void CUDAKernel_GetVector (math::ComplexMatrix* output, uintt length, math::ComplexMatrix* params0, uintt index)
{
  CUDAKernel_getVector (output, length, params0, index);
}

extern "C" __global__ void CUDAKernel_SetIdentity(math::ComplexMatrix* matrix) {
  CUDA_setIdentityMatrix(matrix);
}

extern "C" __global__ void CUDAKernel_SetDiagonal(math::ComplexMatrix* matrix,
                                                  floatt re, floatt im) {
  CUDA_setDiagonalMatrix(matrix, re, im);
}

extern "C" __global__ void CUDAKernel_Invert(math::ComplexMatrix* output,
                                             math::ComplexMatrix* matrix,
                                             math::ComplexMatrix* aux1,
                                             math::ComplexMatrix* aux2,
                                             math::ComplexMatrix* aux3) {
  CUDA_invertMatrix(output, matrix, aux1, aux2, aux3);
}

extern "C" __global__ void CUDAKernel_CompareRe(floatt* sums,
                                                math::ComplexMatrix* matrix1,
                                                math::ComplexMatrix* matrix2,
                                                floatt* buffer,
                                                uint bufferLength)
{
  CUDA_compareReMatrix(sums, matrix1, matrix2, buffer);
}

extern "C" __global__ void CUDAKernel_Compare(floatt* sums,
                                              math::ComplexMatrix* matrix1,
                                              math::ComplexMatrix* matrix2,
                                              floatt* buffer,
                                              uint bufferLength)
{
  CUDA_compare(sums, matrix1, matrix2, buffer);
}

extern "C" __global__ void CUDAKernel_CompareOpt(floatt* sums,
                                                 math::ComplexMatrix* matrix1,
                                                 math::ComplexMatrix* matrix2)
{
  extern __shared__ floatt sharedBufferFloatt[];
  CUDA_compareOpt(sums, matrix1, matrix2, sharedBufferFloatt);
}

extern "C" __global__ void CUDAKernel_CompareOptVer2(floatt* sums,
                                                     math::ComplexMatrix* matrix1,
                                                     math::ComplexMatrix* matrix2)
{
  extern __shared__ floatt sharedBufferFloatt[];
  CUDA_compareOptVer2(sums, matrix1, matrix2, sharedBufferFloatt);
}

extern "C" __global__ void CUDAKernel_MagnitudeOpt(floatt* sums,
                                                   math::ComplexMatrix* matrix)
{
  extern __shared__ floatt bufferFloat[];
  CUDA_magnitudeOpt(sums, matrix, bufferFloat);
}

extern "C" __global__ void CUDAKernel_MagnitudeOptVer2(floatt* sums,
                                                       math::ComplexMatrix* matrix) {
  extern __shared__ floatt bufferFloat[];
  CUDA_magnitudeOptVer2(sums, matrix, bufferFloat);
}

extern "C" __global__ void CUDAKernel_IsUpperTriangular(int* outcome,
                                                        math::ComplexMatrix* matrix) {
  int is = CUDA_isUpperTriangular(matrix);
  (*outcome) = is;
}

extern "C" __global__ void CUDAKernel_Sigmoid(math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) {
  cuda_sigmoid (omatrix, imatrix);
}

extern "C" __global__ void CUDAKernel_DSigmoid (math::ComplexMatrix* matrix) {
  cuda_dsigmoid (matrix, matrix);
}

extern "C" __global__ void CUDAKernel_MultiplyDSigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* matrix) {
  cuda_multiplyDSigmoid (omatrix, matrix);
}

extern "C" __global__ void CUDAKernel_SigmoidDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_sigmoidDim (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DSigmoidDim (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_dsigmoidDim (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_MultiplyDSigmoidDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* matrix, uintt* ex) {
  cuda_multiplyDSigmoidDim (omatrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_SigmoidDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_sigmoidDimPeriodic (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DSigmoidDimPeriodic (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_dsigmoidDimPeriodic (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_MultiplyDSigmoidDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* matrix, uintt* ex) {
  cuda_multiplyDSigmoidDimPeriodic (omatrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_Tanh(math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) {
  cuda_tanh (omatrix, imatrix);
}

extern "C" __global__ void CUDAKernel_DTanh (math::ComplexMatrix* matrix) {
  cuda_dtanh (matrix, matrix);
}

extern "C" __global__ void CUDAKernel_TanhDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_tanhDim (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DTanhDim (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_dtanhDim (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_TanhDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_tanhDimPeriodic (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DTanhDimPeriodic (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_dtanhDimPeriodic (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_Sin(math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) {
  cuda_sin (omatrix, imatrix);
}

extern "C" __global__ void CUDAKernel_MultiplyDSin (math::ComplexMatrix* matrix) {
  cuda_multiplyDSin (matrix, matrix);
}

extern "C" __global__ void CUDAKernel_DSin (math::ComplexMatrix* matrix) {
  cuda_dsin (matrix, matrix);
}

extern "C" __global__ void CUDAKernel_SinDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_sinDim (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_MultiplyDSinDim (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_multiplyDSinDim (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_DSinDim (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_dsinDim (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_SinDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_sinDimPeriodic (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_MultiplyDSinDimPeriodic (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_multiplyDSinDimPeriodic (matrix, matrix, ex);
}

extern "C" __global__ void CUDAKernel_DSinDimPeriodic (math::ComplexMatrix* matrix, uintt* ex) {
  cuda_dsinDimPeriodic (matrix, matrix, ex);
}

// Relu
extern "C" __global__ void CUDAKernel_Relu (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) {
  cuda_relu (omatrix, imatrix);
}

extern "C" __global__ void CUDAKernel_DRelu (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1) {
  cuda_drelu (matrix, matrix1);
}

extern "C" __global__ void CUDAKernel_ReluDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_reluDim (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DReluDim (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1,  uintt* ex) {
  cuda_dreluDim (matrix, matrix1, ex);
}

extern "C" __global__ void CUDAKernel_ReluDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_reluDimPeriodic (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DReluDimPeriodic (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt* ex) {
  cuda_dreluDimPeriodic (matrix, matrix1, ex);
}

// PRelu
extern "C" __global__ void CUDAKernel_PRelu (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) {
  cuda_prelu (omatrix, imatrix);
}

extern "C" __global__ void CUDAKernel_DPRelu (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1) {
  cuda_dprelu (matrix, matrix1);
}

extern "C" __global__ void CUDAKernel_PReluDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_preluDim (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DPReluDim (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt* ex) {
  cuda_dpreluDim (matrix, matrix1, ex);
}

extern "C" __global__ void CUDAKernel_PReluDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_preluDimPeriodic (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DPReluDimPeriodic (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt* ex) {
  cuda_dpreluDimPeriodic (matrix, matrix1, ex);
}

// Softplus
extern "C" __global__ void CUDAKernel_Softplus (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) {
  cuda_softplus (omatrix, imatrix);
}

extern "C" __global__ void CUDAKernel_DSoftplus (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1) {
  cuda_dsoftplus (matrix, matrix1);
}

extern "C" __global__ void CUDAKernel_SoftplusDim (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_softplusDim (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DSoftplusDim (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt* ex) {
  cuda_dsoftplusDim (matrix, matrix1, ex);
}

extern "C" __global__ void CUDAKernel_SoftplusDimPeriodic (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, uintt* ex) {
  cuda_softplusDimPeriodic (omatrix, imatrix, ex);
}

extern "C" __global__ void CUDAKernel_DSoftplusDimPeriodic (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt* ex) {
  cuda_dsoftplusDimPeriodic (matrix, matrix1, ex);
}

extern "C" __global__ void CUDAKernel_CrossEntropy (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_crossEntropy (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_CrossEntropyRe (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_crossEntropyRe (output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SumShared (floatt* rebuffer, floatt* imbuffer, math::ComplexMatrix* params0)
{
  CUDA_sumShared (rebuffer, imbuffer, params0);
}

extern "C" __global__ void CUDAKernel_CalculateTriangularH(
    math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* R,
    math::ComplexMatrix* aux1, math::ComplexMatrix* aux2, math::ComplexMatrix* aux3,
    math::ComplexMatrix* aux4, math::ComplexMatrix* aux5, math::ComplexMatrix* aux6)
{
  CUDA_calcUTMatrix_GR (H, Q, R, aux1, aux2, aux3, aux4, aux5, aux6);
}

extern "C" __global__ void CUDAKernel_CalculateTriangularHStep(
    math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* R,
    math::ComplexMatrix* aux1, math::ComplexMatrix* aux2, math::ComplexMatrix* aux3,
    math::ComplexMatrix* aux4, math::ComplexMatrix* aux5, math::ComplexMatrix* aux6)
{
  CUDA_calcUTMatrixStep_GR (H, Q, R, aux1, aux2, aux3, aux4, aux5, aux6);
}

extern "C" __global__ void
CUDAKernel_TensorProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_tensorProduct (output, params0, params1);
}

extern "C" __global__ void
CUDAKernel_TensorProductDim (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt* ex)
{
  CUDA_tensorProductDim (output, params0, params1, ex);
}

extern "C" __global__ void
CUDAKernel_HadamardProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_hadamardProduct (output, params0, params1);
}

extern "C" __global__ void
CUDAKernel_PHadamardProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  CUDA_phadamardProduct (output, params0, params1);
}

extern "C" __global__ void
CUDAKernel_Convolve (math::ComplexMatrix* output, math::ComplexMatrix* matrix, math::ComplexMatrix* kernel)
{
  CudaKernel_convolve (output, matrix, kernel);
}

extern "C" __global__ void
CUDAKernel_PoolAverage (math::ComplexMatrix* output, math::ComplexMatrix* matrix, uintt* ex)
{
  extern __shared__ floatt cache[];
  CUDA_poolAverage (output, matrix, ex, cache);
}

extern "C" __global__ void
CUDAKernel_GenericApi_AddConst (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, floatt params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_AddConst (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Add (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_Add (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Subtract (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_Subtract (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DotProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_DotProduct (outputs, params1, params2, mapper);
}

  extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyConst (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, floatt params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_MultiplyConst (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_HadamardProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_HadamardProduct (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_PartialHadamardProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_PartialHadamardProduct (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_PHadamardProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_PartialHadamardProduct (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_TensorProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_TensorProduct (outputs, params1, params2, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Transpose (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_Transpose (outputs, params1, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Tanh (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_tanh (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DTanh (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_dtanh (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDTanh (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_multiplyDTanh (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Sin (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_sin (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DSin (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_dsin (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDSin (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_multiplyDSin (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Sigmoid (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_sigmoid (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DSigmoid (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_dsigmoid (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDSigmoid (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_multiplyDSigmoid (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Softplus (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_softplus (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DSoftplus (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_dsoftplus (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDSoftplus (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_multiplyDSoftplus (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Relu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_relu (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DRelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_drelu (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDRelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_multiplyDRelu (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Prelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_prelu (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DPrelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_dprelu (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDPrelu (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_multiplyDPrelu (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_Linear (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_linear (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_DLinear (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_dlinear (outputs, params, mapper);
}

extern "C" __global__ void
CUDAKernel_GenericApi_MultiplyDLinear (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params, oap::ThreadsMapperS* mapper) {
  cuda_genericApi_multiplyDLinear (outputs, params, mapper);
}

