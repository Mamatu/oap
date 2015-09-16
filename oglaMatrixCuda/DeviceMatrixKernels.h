#ifndef DEVICEMATRIXKERNELS_H
#define DEVICEMATRIXKERNELS_H

#include "Matrix.h"
#include "KernelExecutor.h"

CUresult DEVICEKernel_DotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1, cuda::Kernel& kernel);

CUresult DEVICEKernel_Transpose(math::Matrix* output, math::Matrix* params0,
                                cuda::Kernel& kernel);

CUresult DEVICEKernel_SetIdentity(math::Matrix* matrix, cuda::Kernel& kernel);

#endif  // DEVICEKERNELS_H
