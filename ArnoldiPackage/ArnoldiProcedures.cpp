#include "ArnoldiProcedures.h"

void CuProcedure_CalculateTriangularH(void** params,
        ::cuda::Kernel& kernel, void* image) {
    cuda::Kernel::ExecuteKernel("CUDAKernel_CalculateTriangularH",
            params, kernel, image);
}

void CuProcedure_CalculateH(void** params,
        ::cuda::Kernel& kernel, void* image) {
    cuda::Kernel::ExecuteKernel("CUDAKernel_CalculateH", params, kernel, image);
}