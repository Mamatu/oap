#ifndef DEVICEMATRIXKERNELS_H
#define DEVICEMATRIXKERNELS_H

#include "Matrix.h"
#include "KernelExecutor.h"

CUresult DEVICEKernel_DotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1, device::Kernel& kernel);

CUresult DEVICEKernel_Transpose(math::Matrix* output, math::Matrix* params0,
                                device::Kernel& kernel);

CUresult DEVICEKernel_SetIdentity(math::Matrix* matrix, device::Kernel& kernel);

CUresult DEVICEKernel_Substract(math::Matrix* output, math::Matrix* params0,
                                math::Matrix* params1, device::Kernel& kernel);

CUresult DEVICEKernel_CalcTriangularH(math::Matrix* H1, math::Matrix* Q,
                                      math::Matrix* R1, math::Matrix* Q1,
                                      math::Matrix* QJ, math::Matrix* Q2,
                                      math::Matrix* R2, math::Matrix* G,
                                      math::Matrix* GT, uintt columns,
                                      uintt rows, device::Kernel& kernel);

#endif  // DEVICEKERNELS_H
