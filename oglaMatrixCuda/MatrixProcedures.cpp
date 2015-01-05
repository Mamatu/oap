/* 
 * File:   MatrixProcedures.cpp
 * Author: mmatula
 * 
 * Created on January 3, 2015, 8:37 PM
 */

#include "MatrixProcedures.h"
#include "KernelExecutor.h"
#include "DebugLogs.h"
#include "CudaUtils.h"
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

void CuMatrix_dotProduct(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    CuMatrixModule::Init();
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    void* params[] = {&output, &params0, &params1};
    CuMatrixModule::GetInstance().execute("CUDAKernel_DotProduct",
        w, h, params, 0);
}

void CuMatrix_dotProductEx(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    MatrixEx* matrixEx) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0, &params1, &matrixEx};
    const uintt w = CudaUtils::GetColumns(matrixEx);
    const uintt h = CudaUtils::GetRows(matrixEx);
    CuMatrixModule::GetInstance().execute("CUDAKernel_DotProductEx",
        w, h, params, 0);
}

void CuMatrix_transposeMatrixEx(math::Matrix* output,
    math::Matrix* params0, MatrixEx* matrixEx) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0, &matrixEx};
    const uintt w = CudaUtils::GetColumns(matrixEx);
    const uintt h = CudaUtils::GetRows(matrixEx);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_TransposeEx", w, h, params, 0);
}

void CuMatrix_transposeMatrix(math::Matrix* output,
    math::Matrix* params0) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Transpose", w, h, params, 0);
}

void CuMatrix_substractMatrix(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0, &params1};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Substract", w, h, params, 0);
}

void CuMatrix_addMatrix(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0, &params1};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Add", w, h, params, 0);
}

void CuMatrix_setVector(math::Matrix* V, uintt index,
    math::Matrix* v, uintt length) {
    CuMatrixModule::Init();
    const uintt w = CudaUtils::GetColumns(v);
    const uintt h = CudaUtils::GetRows(v);
    void* params[] = {&V, &index, &v, &length};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_SetVector", w, h, params, 0);
}

void CuMatrix_magnitude(floatt& output, math::Matrix* param0) {
    CuMatrixModule::Init();
    floatt* doutput = CudaUtils::AllocDeviceObj<floatt>(0);
    const uintt w = CudaUtils::GetColumns(param0);
    const uintt h = CudaUtils::GetRows(param0);
    floatt* buffer = reinterpret_cast<floatt*>
        (CudaUtils::AllocDeviceMem(sizeof (floatt) * w * h / 2));
    void* params[] = {&doutput, &param0, &buffer};
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_Magnitude", w, h, params, 0);
    CudaUtils::CopyDeviceToHost(&output, doutput, sizeof (floatt));
    CudaUtils::FreeDeviceObj(doutput);
    CudaUtils::FreeDeviceMem(buffer);
}

void CuMatrix_setDiagonalMatrix(math::Matrix* v, floatt* re, floatt* im) {
    CuMatrixModule::Init();
    debugAssert("not implemented" == NULL);
}

void CuMatrix_setIdentity(math::Matrix* matrix) {
    CuMatrixModule::Init();
    void* params[] = {&matrix};
    const uintt w = CudaUtils::GetColumns(matrix);
    const uintt h = CudaUtils::GetRows(matrix);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_SetIdentity", w, h, params, 0);
}

void CuMatrix_setZeroMatrix(math::Matrix* matrix) {
    CuMatrixModule::Init();
    debugAssert("not implemented" == NULL);
}

void CuMatrix_QR(math::Matrix* Q,
    math::Matrix* R, math::Matrix* H,
    math::Matrix* R1, math::Matrix* Q1,
    math::Matrix* G, math::Matrix * GT) {
    CuMatrixModule::Init();
    debugAssert("not implemented" == NULL);
}

void CuMatrix_multiplyConstantMatrix(math::Matrix* output,
    math::Matrix* params0, floatt re) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0, &re};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_MultiplyConstantRe", w, h, params, 0);
}

void CuMatrix_multiplyConstantMatrix(math::Matrix* output,
    math::Matrix* params0, floatt re, floatt im) {
    CuMatrixModule::Init();
    void* params[] = {&output, &params0, &re, &im};
    const uintt w = CudaUtils::GetColumns(output);
    const uintt h = CudaUtils::GetRows(output);
    CuMatrixModule::GetInstance().execute(
        "CUDAKernel_MultiplyConstantReal", w, h, params, 0);

}

void CuMatrix_getVector(math::Matrix* vector, uintt rows,
    math::Matrix* matrix, uintt column) {
}