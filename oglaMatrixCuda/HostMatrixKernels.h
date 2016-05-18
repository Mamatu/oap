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
