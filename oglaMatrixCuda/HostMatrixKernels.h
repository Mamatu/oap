#ifndef HOSTMATRIXKERNELS_H
#define HOSTMATRIXKERNELS_H

#include "Matrix.h"
#include "KernelExecutor.h"

CUresult HOSTKernel_QRGR(math::Matrix* output0, math::Matrix* output1,
                     math::Matrix* params0, math::Matrix* aux0,
                     math::Matrix* aux1, math::Matrix* aux2,
                     math::Matrix* aux3, device::Kernel& kernel);

#endif  // HOSTMATRIXKERNELS_H
