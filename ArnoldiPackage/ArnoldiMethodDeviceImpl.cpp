#include <math.h>
#include "ArnoldiMethodDeviceImpl.h"
#include "ArnoldiProcedures.h"

#ifdef DEBUG
#define PRINT_STATUS(d) if(d!=0) { fprintf(stderr,"Status == %d \n",d); abort();}
#else
#define PRINT_STATUS(d) d
#endif
#define MIN_VALUE 0.001


namespace math {

ArnoldiMethodGpu::ArnoldiMethodGpu() :
IArnoldiMethod(DeviceMatrixModules::GetInstance()) {
    this->m_rho = 1. / 3.14;
    this->m_k = 0;
    this->m_wantedCount = 0;
}

ArnoldiMethodGpu::ArnoldiMethodGpu(
    MatrixModule* matrixModule) :
IArnoldiMethod(matrixModule) {
    this->m_rho = 1. / 3.14;
    this->m_k = 0;
    this->m_wantedCount = 0;
}

void ArnoldiMethodGpu::setHSize(uintt k) {
    this->m_k = k;
}

void ArnoldiMethodGpu::setRho(floatt rho) {
    this->m_rho = rho;
}

ArnoldiMethodGpu::~ArnoldiMethodGpu() {
    // not implemented
}

void ArnoldiMethodGpu::execute() {
    m_wantedCount = m_count;
    outputs.realColumns = m_wantedCount;
    outputs.columns = m_wantedCount;
    outputs.realRows = 1;
    outputs.rows = 1;
    outputs.imValues = NULL;
    outputs.reValues = m_reoutputs;
    cuHArnoldi.execute(&outputs, m_matrix, m_k, m_wantedCount);
    for (uintt fa = 0; fa < m_wantedCount; ++fa) {
        debug("eig[%u] = %f", fa, outputs.reValues[fa]);
    }
#if 0    
    debugFunc();
    m_image = ::cuda::Kernel::LoadImage(kernelsFiles);
    debugFunc();
    alloc(m_matrix);
    debugFunc();

    floatt a[] = {209876.114322, 5543.454862, -1.931923, 150.393653,
        5545.494910, 204558.192376, 5615.605322, 152.459566,
        0.000000, 5615.551427, 209721.857008, 220.557474,
        0.000000, 0.000000, 72.205295, 204259.343999};

    cuda::CopyHostMatrixToDeviceMatrix(m_A.m_matrix, m_matrix);
    m_kernel.setSharedMemory(0);
    m_kernel.setThreadsCount(m_matrix->columns, m_matrix->rows);
    void* params[] = {
        &H.m_matrix, &m_A.m_matrix,
        &w.m_matrix, &v.m_matrix,
        &f.m_matrix, &V.m_matrix, &transposeV.m_matrix,
        &s.m_matrix, &vs.m_matrix,
        &h.m_matrix, &vh.m_matrix
    };
    CuProcedure_CalculateH(params, m_kernel, m_image);
    m_kernel.setSharedMemory(0);
    m_kernel.setThreadsCount(m_k, m_k);
    void* params1[] = {
        &H.m_matrix, &Q.m_matrix, &R1.m_matrix,
        &Q1.m_matrix, &QJ.m_matrix, &Q2.m_matrix,
        &R2.m_matrix, &G.m_matrix, &GT.m_matrix
    };
    CuProcedure_CalculateTriangularH(params1, m_kernel, m_image);
#endif
    //cuda::Kernel::ExecuteKernel("CUDAKernel_Execute", params1, m_kernel, m_image);
    debugFunc();
}

}
