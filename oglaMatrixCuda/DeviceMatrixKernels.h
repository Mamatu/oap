#ifndef DEVICEMATRIXKERNELS_H
#define DEVICEMATRIXKERNELS_H

#include "Matrix.h"
#include "KernelExecutor.h"

CUresult DEVICEKernel_DotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1, device::Kernel& kernel);

CUresult DEVICEKernel_Transpose(math::Matrix* output, math::Matrix* params0,
                                device::Kernel& kernel);

CUresult DEVICEKernel_SetIdentity(math::Matrix* matrix, device::Kernel& kernel);

#endif  // DEVICEKERNELS_H
