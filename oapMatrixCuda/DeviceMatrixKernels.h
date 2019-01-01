/*
 * Copyright 2016 - 2018 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef DEVICEMATRIXKERNELS_H
#define DEVICEMATRIXKERNELS_H

#include "Matrix.h"
#include "KernelExecutor.h"

bool DEVICEKernel_DotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1, oap::cuda::Kernel& kernel);

bool DEVICEKernel_Transpose(math::Matrix* output, math::Matrix* params0,
                                oap::cuda::Kernel& kernel);

bool DEVICEKernel_SetIdentity(math::Matrix* matrix, oap::cuda::Kernel& kernel);

bool DEVICEKernel_Substract(math::Matrix* output, math::Matrix* params0,
                                math::Matrix* params1, oap::cuda::Kernel& kernel);

bool DEVICEKernel_CalcTriangularH(math::Matrix* H1, math::Matrix* Q,
                                      math::Matrix* R1, math::Matrix* Q1,
                                      math::Matrix* QJ, math::Matrix* Q2,
                                      math::Matrix* R2, math::Matrix* G,
                                      math::Matrix* GT, uintt columns,
                                      uintt rows, oap::cuda::Kernel& kernel);

#endif  // DEVICEKERNELS_H
