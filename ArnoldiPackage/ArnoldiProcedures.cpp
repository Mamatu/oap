#include "ArnoldiProcedures.h"

void CUDAKernel_CalculateHQ(void** params, ::cuda::Kernel& kernel, void* image) {
    cuda::Kernel::ExecuteKernel("CUDAKernel_CalculateHQ", params, kernel, image);
}