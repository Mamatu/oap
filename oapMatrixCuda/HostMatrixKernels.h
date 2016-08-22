/*
 * Copyright 2016 Marcin Matula
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



#ifndef HOSTMATRIXKERNELS_H
#define HOSTMATRIXKERNELS_H

#include "Matrix.h"
#include "KernelExecutor.h"
#include "MatrixProcedures.h"

CUresult HOSTKernel_QRGR(math::Matrix* output0, math::Matrix* output1,
                         math::Matrix* params0, math::Matrix* aux0,
                         math::Matrix* aux1, math::Matrix* aux2,
                         math::Matrix* aux3, device::Kernel& kernel);

void HOSTKernel_CalcTriangularH(math::Matrix* H1, math::Matrix* Q,
                                math::Matrix* R1, math::Matrix* Q1,
                                math::Matrix* QJ, math::Matrix* Q2,
                                math::Matrix* R2, math::Matrix* G,
                                math::Matrix* GT, CuMatrix& m_cuMatrix);

#endif  // HOSTMATRIXKERNELS_H
