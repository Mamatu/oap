/*
 * Copyright 2016 - 2021 Marcin Matula
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

bool DEVICEKernel_DotProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                 math::ComplexMatrix* params1, oap::cuda::Kernel& kernel);

bool DEVICEKernel_Transpose(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                oap::cuda::Kernel& kernel);

bool DEVICEKernel_SetIdentity(math::ComplexMatrix* matrix, oap::cuda::Kernel& kernel);

bool DEVICEKernel_Substract(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                math::ComplexMatrix* params1, oap::cuda::Kernel& kernel);

bool DEVICEKernel_CalcTriangularH(math::ComplexMatrix* H1, math::ComplexMatrix* Q,
                                      math::ComplexMatrix* R1, math::ComplexMatrix* Q1,
                                      math::ComplexMatrix* QJ, math::ComplexMatrix* Q2,
                                      math::ComplexMatrix* R2, math::ComplexMatrix* G,
                                      math::ComplexMatrix* GT, uintt columns,
                                      uintt rows, oap::cuda::Kernel& kernel);

#endif  // DEVICEKERNELS_H
