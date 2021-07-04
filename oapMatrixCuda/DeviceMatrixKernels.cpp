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

#include "DeviceMatrixKernels.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "ThreadsMapper.hpp"

void prepareDims(uintt w, uintt h, oap::cuda::Kernel& kernel) {
  uint blocks[2];
  uint threads[2];
  uintt maxThreadsPerBlock = kernel.getMaxThreadsPerBlock();
  oap::utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
  kernel.setBlocksCount(blocks[0], blocks[1]);
  kernel.setThreadsCount(threads[0], threads[1]);
}

bool execute(const char* functionName, math::ComplexMatrix* matrix, void** params,
                 uintt sharedMemory, oap::cuda::Kernel& kernel) {
  uintt w = oap::cuda::GetColumns(matrix);
  uintt h = oap::cuda::GetRows(matrix);
  prepareDims(w, h, kernel);
  kernel.setSharedMemory(sharedMemory);
  return ::oap::cuda::Kernel::Execute(functionName, const_cast<const void**>(params), kernel);
}

bool DEVICEKernel_DotProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                 math::ComplexMatrix* params1,
                                 oap::cuda::Kernel& kernel) {
  void* params[] = {&output, &params0, &params1};
  return execute("CUDAKernel_DotProduct", output, params, 0, kernel);
}

bool DEVICEKernel_Transpose(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                oap::cuda::Kernel& kernel) {
  void* params[] = {&output, &params0};
  return execute("CUDAKernel_Transpose", output, params, 0, kernel);
}

bool DEVICEKernel_SetIdentity(math::ComplexMatrix* matrix,
                                  oap::cuda::Kernel& kernel) {
  void* params[] = {&matrix};
  return execute("CUDAKernel_SetIdentity", matrix, params, 0, kernel);
}

bool DEVICEKernel_Substract(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                math::ComplexMatrix* params1, oap::cuda::Kernel& kernel) {
  void* params[] = {&output, &params0, &params1};
  return execute("CUDAKernel_Substract", output, params, 0, kernel);
}

bool DEVICEKernel_CalcTriangularH(math::ComplexMatrix* H1, math::ComplexMatrix* Q,
                                      math::ComplexMatrix* R1, math::ComplexMatrix* Q1,
                                      math::ComplexMatrix* QJ, math::ComplexMatrix* Q2,
                                      math::ComplexMatrix* R2, math::ComplexMatrix* G,
                                      math::ComplexMatrix* GT, uintt columns,
                                      uintt rows, oap::cuda::Kernel& kernel) {
  const void* params[] = {&H1, &Q, &R1, &Q1, &QJ, &Q2, &R2, &G, &GT};
  kernel.setDimensions(columns, rows);
  return oap::cuda::Kernel::Execute("CUDAKernel_CalculateTriangularH", params, kernel);
}
