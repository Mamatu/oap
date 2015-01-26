/* 
 * File:   MatrixProcedures.cpp
 * Author: mmatula
 * 
 * Created on January 3, 2015, 8:37 PM
 */

#include "MatrixProcedures.h"
#include "DebugLogs.h"
#include "ThreadsMapper.h"

CUresult CuMatrix::execute(const char* functionName,
    uintt w, uintt h,
    void** params,
    uintt sharedMemory) {
    uintt blocks[2];
    uintt threads[2];
    utils::mapper::SetThreadsBlocks(blocks, threads,
        w, h, m_maxThreadsPerBlock);
    m_kernel.setBlocksCount(blocks[0], blocks[1]);
    m_kernel.setThreadsCount(threads[0], threads[1]);
    m_kernel.setSharedMemory(sharedMemory);
    return ::cuda::Kernel::Execute(functionName, params, m_kernel, m_image);
}

CuMatrix::CuMatrix() : m_cuResult(CUDA_SUCCESS),
    m_dcompareOutputBuffer(CuMatrix::CUDA),
    m_hcompareOutputBuffer(CuMatrix::HOST),
    m_magnitudeBuffer(CuMatrix::CUDA) {
    m_pathes[0] = "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.cubin";
    m_pathes[1] = "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/albert/liboglaMatrixCuda.cubin";
    m_pathes[2] = NULL;
    init();
    m_magniuteOutput = CudaUtils::AllocDeviceObj<floatt>(0);
}

void CuMatrix::init() {
    if (!m_isIntialized) {
        m_isIntialized = true;
        cuda::Context::Instance().init();
        m_image = ::cuda::Kernel::LoadImage(m_pathes);
        CUdevprop devprop;
        m_kernel.getDeviceProperties(devprop);
        m_maxThreadsPerBlock = devprop.maxThreadsPerBlock;
    }
}

CuMatrix::~CuMatrix() {
    CudaUtils::FreeDeviceObj(m_magniuteOutput);
    if (m_isIntialized) {
        cuda::Kernel::FreeImage(m_image);
        cuda::Context::Instance().destroy();
    }
}

void CuMatrix::dotProduct(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    void* params[] = {&output, &params0, &params1};
    m_cuResult = execute("CUDAKernel_DotProduct",
        w, h, params, 0);
}

void CuMatrix::dotProductEx(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    MatrixEx* matrixEx) {
    void* params[] = {&output, &params0, &params1, &matrixEx};
    const uintt w = CudaUtils::GetColumns(matrixEx);
    const uintt h = CudaUtils::GetRows(matrixEx);
    m_cuResult = execute("CUDAKernel_DotProductEx",
        w, h, params, 0);
}

void CuMatrix::transposeMatrixEx(math::Matrix* output,
    math::Matrix* params0, MatrixEx* matrixEx) {
    void* params[] = {&output, &params0, &matrixEx};
    const uintt w = CudaUtils::GetColumns(matrixEx);
    const uintt h = CudaUtils::GetRows(matrixEx);
    execute(
        "CUDAKernel_TransposeEx", w, h, params, 0);
}

void CuMatrix::transposeMatrix(math::Matrix* output,
    math::Matrix* params0) {
    void* params[] = {&output, &params0};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    m_cuResult = execute(
        "CUDAKernel_Transpose", w, h, params, 0);
}

void CuMatrix::substract(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    void* params[] = {&output, &params0, &params1};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    m_cuResult = execute(
        "CUDAKernel_Substract", w, h, params, 0);
}

void CuMatrix::addMatrix(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    void* params[] = {&output, &params0, &params1};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    m_cuResult = execute(
        "CUDAKernel_Add", w, h, params, 0);
}

void CuMatrix::setVector(math::Matrix* V, uintt column,
    math::Matrix* v, uintt length) {
    const uintt w = CudaUtils::GetColumns(v);
    const uintt h = CudaUtils::GetRows(v);
    void* params[] = {&V, &column, &v, &length};
    m_cuResult = execute(
        "CUDAKernel_SetVector", w, h, params, 0);
}

void CuMatrix::getVector(math::Matrix* vector, uintt length,
    math::Matrix* matrix, uintt column) {
    const uintt w = CudaUtils::GetColumns(vector);
    const uintt h = CudaUtils::GetRows(vector);
    void* params[] = {&vector, &length, &matrix, &column};
    m_cuResult = execute("CUDAKernel_GetVector", w, h, params, 0);
}

void CuMatrix::magnitude(floatt& output, math::Matrix* param0) {
    const uintt w = CudaUtils::GetColumns(param0);
    const uintt h = CudaUtils::GetRows(param0);
    m_magnitudeBuffer.realloc(sizeof (floatt) * w * h / 2);
    void* params[] = {&m_magniuteOutput, &param0, &m_magnitudeBuffer.m_buffer};
    m_cuResult = execute("CUDAKernel_Magnitude", w, h, params, 0);
    CudaUtils::CopyDeviceToHost(&output, m_magniuteOutput, sizeof (floatt));
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

void CuMatrix::QR(math::Matrix* Q,
    math::Matrix* R, math::Matrix* H,
    math::Matrix* aux0, math::Matrix* aux1,
    math::Matrix* aux2, math::Matrix * aux3) {
    void* params[] = {
        &Q, &R, &H,
        &aux0, &aux1,
        &aux2, &aux3
    };
    const uintt w = CudaUtils::GetColumns(H);
    const uintt h = CudaUtils::GetRows(H);
    m_cuResult = execute("CUDAKernel_QR", w, h, params, 0);
}

void CuMatrix::multiplyConstantMatrix(math::Matrix* output,
    math::Matrix* params0, floatt re) {
    void* params[] = {&output, &params0, &re};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    m_cuResult = execute("CUDAKernel_MultiplyConstantRe", w, h, params, 0);
}

void CuMatrix::multiplyConstantMatrix(math::Matrix* output,
    math::Matrix* params0, floatt re, floatt im) {
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
    uintt size = w * h * sizeof (int) / 2;
    uintt blocks[2];
    uintt threads[2];
    m_kernel.setThreadsBlocks(blocks, threads, w, h);
    m_kernel.setBlocksCount(blocks[0], blocks[1]);
    m_kernel.setThreadsCount(threads[0], threads[1]);
    m_kernel.setSharedMemory(size);

    m_dcompareOutputBuffer.realloc(sizeof (int) * blocks[0] * blocks[1]);
    m_hcompareOutputBuffer.realloc(sizeof (int) * blocks[0] * blocks[1]);

    void* params[] = {&m_dcompareOutputBuffer.m_buffer, &matrix1, &matrix2};

    m_cuResult = ::cuda::Kernel::Execute("CUDAKernel_CompareOpt", params, m_kernel, m_image);

    CudaUtils::CopyDeviceToHost(m_hcompareOutputBuffer.m_buffer,
        m_dcompareOutputBuffer.m_buffer, sizeof (int) * blocks[0] * blocks[1]);

    uintt outcome = 0;
    for (uint fa = 0; fa < blocks[0] * blocks[1]; ++fa) {
        outcome += m_hcompareOutputBuffer.m_buffer[fa];
    }
    return outcome == w * h;
}

CUresult CuMatrix::getStatus() const {
    return m_cuResult;
}
