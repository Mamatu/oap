/* 
 * File:   MatrixProcedures.cpp
 * Author: mmatula
 * 
 * Created on January 3, 2015, 8:37 PM
 */

#include "MatrixProcedures.h"
#include "KernelExecutor.h"
#include "DebugLogs.h"
#include "ThreadsMapper.h"

class CuMatrixModule {
    static CuMatrixModule m_cuMatrixModule;
    bool m_isIntialized;
    ::cuda::Kernel m_kernel;
    const char* m_pathes[3];
    void* m_image;
    uintt m_maxThreadsPerBlock;
public:
    static CuMatrixModule& GetInstance();
    static void Init();

    CuMatrixModule();
    ~CuMatrixModule();
    void init();
    void execute(const char* functionName,
        uintt w, uintt h,
        void** params,
        uintt sharedMemory = 0);
};

CuMatrixModule::CuMatrixModule() :
m_isIntialized(false) {
    m_pathes[0] = "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.cubin";
    m_pathes[1] = "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/albert/liboglaMatrixCuda.cubin";
    m_pathes[2] = NULL;
}

CuMatrixModule::~CuMatrixModule() {
    if (m_isIntialized) {
        ::cuda::Kernel::FreeImage(m_image);
    }
}

void CuMatrixModule::execute(const char* functionName,
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
    ::cuda::Kernel::ExecuteKernel(functionName, params, m_kernel, m_image);
}

void CuMatrixModule::init() {
    if (!m_isIntialized) {
        m_isIntialized = true;
        m_image = ::cuda::Kernel::LoadImage(m_pathes);
        CUdevprop devprop;
        m_kernel.getDeviceProperties(devprop);
        m_maxThreadsPerBlock = devprop.maxThreadsPerBlock;
    }
}

CuMatrixModule CuMatrixModule::m_cuMatrixModule;

CuMatrixModule& CuMatrixModule::GetInstance() {
    return m_cuMatrixModule;
}

void CuMatrixModule::Init() {
    GetInstance().init();
}

CuMatrix::CuMatrix() {
    cuda::Context::Instance().init();
    CuMatrixModule::Init();
    m_magniuteOutput = CudaUtils::AllocDeviceObj<floatt>(0);
    m_dcompareOutput = CudaUtils::AllocDeviceObj<uintt>(0);
}

CuMatrix::~CuMatrix() {
    CudaUtils::FreeDeviceObj(m_magniuteOutput);
    CudaUtils::FreeDeviceObj(m_dcompareOutput);
}

void CuMatrix::dotProduct(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    void* params[] = {&output, &params0, &params1};
    CuMatrixModule::GetInstance().execute("CUDAKernel_DotProduct",
        w, h, params, 0);
}

void CuMatrix::dotProductEx(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    MatrixEx* matrixEx) {
    void* params[] = {&output, &params0, &params1, &matrixEx};
    const uintt w = CudaUtils::GetColumns(matrixEx);
    const uintt h = CudaUtils::GetRows(matrixEx);
    CuMatrixModule::GetInstance().execute("CUDAKernel_DotProductEx",
        w, h, params, 0);
}

void CuMatrix::transposeMatrixEx(math::Matrix* output,
    math::Matrix* params0, MatrixEx* matrixEx) {
    void* params[] = {&output, &params0, &matrixEx};
    const uintt w = CudaUtils::GetColumns(matrixEx);
    const uintt h = CudaUtils::GetRows(matrixEx);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_TransposeEx", w, h, params, 0);
}

void CuMatrix::transposeMatrix(math::Matrix* output,
    math::Matrix* params0) {
    void* params[] = {&output, &params0};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Transpose", w, h, params, 0);
}

void CuMatrix::substract(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    void* params[] = {&output, &params0, &params1};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Substract", w, h, params, 0);
}

void CuMatrix::addMatrix(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    void* params[] = {&output, &params0, &params1};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Add", w, h, params, 0);
}

void CuMatrix::setVector(math::Matrix* V, uintt column,
    math::Matrix* v, uintt length) {
    const uintt w = CudaUtils::GetColumns(v);
    const uintt h = CudaUtils::GetRows(v);
    void* params[] = {&V, &column, &v, &length};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_SetVector", w, h, params, 0);
}

void CuMatrix::getVector(math::Matrix* vector, uintt length,
    math::Matrix* matrix, uintt column) {
    const uintt w = CudaUtils::GetColumns(vector);
    const uintt h = CudaUtils::GetRows(vector);
    void* params[] = {&vector, &length, &matrix, &column};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_GetVector", w, h, params, 0);
}

void CuMatrix::magnitude(floatt& output, math::Matrix* param0) {
    const uintt w = CudaUtils::GetColumns(param0);
    const uintt h = CudaUtils::GetRows(param0);
    m_magnitudeBuffer.realloc(sizeof (floatt) * w * h / 2);
    void* params[] = {&m_magniuteOutput, &param0, &m_magnitudeBuffer.m_buffer};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Magnitude", w, h, params, 0);
    CudaUtils::CopyDeviceToHost(&output, m_magniuteOutput, sizeof (floatt));
}

void CuMatrix::setDiagonal(math::Matrix* matrix, floatt re, floatt im) {
    const uintt w = CudaUtils::GetColumns(matrix);
    const uintt h = CudaUtils::GetRows(matrix);
    void* params[] = {&matrix, &re, &im};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_SetDiagonal", w, h, params, 0);
}

void CuMatrix::setIdentity(math::Matrix* matrix) {
    void* params[] = {&matrix};
    const uintt w = CudaUtils::GetColumns(matrix);
    const uintt h = CudaUtils::GetRows(matrix);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_SetIdentity", w, h, params, 0);
}

void CuMatrix::setZeroMatrix(math::Matrix* matrix) {
    CudaUtils::SetZeroMatrix(matrix, true, true);
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
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_QR", w, h, params, 0);
}

void CuMatrix::multiplyConstantMatrix(math::Matrix* output,
    math::Matrix* params0, floatt re) {
    void* params[] = {&output, &params0, &re};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_MultiplyConstantRe", w, h, params, 0);
}

void CuMatrix::multiplyConstantMatrix(math::Matrix* output,
    math::Matrix* params0, floatt re, floatt im) {
    void* params[] = {&output, &params0, &re, &im};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_MultiplyConstant", w, h, params, 0);
}

bool CuMatrix::compare(math::Matrix* matrix1, math::Matrix* matrix2) {
    if (matrix1 == matrix2) {
        return true;
    }
    const uintt w = CudaUtils::GetColumns(matrix1);
    const uintt h = CudaUtils::GetRows(matrix1);
    uintt size = w * h * sizeof (int) / 2;
    m_compareBuffer.realloc(size);
    void* params[] = {&m_dcompareOutput, &matrix1, &matrix2, &m_compareBuffer.m_buffer};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Compare", w, h, params, 0);
    uintt hsum;
    CudaUtils::CopyDeviceToHost(&hsum, m_dcompareOutput, sizeof (uintt));

    return hsum == w * h;
}
