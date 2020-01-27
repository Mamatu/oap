/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "DeviceMatrixKernels.h"
#include "oapCudaMatrixUtils.h"
#include "ThreadsMapper.h"

void prepareDims(uintt w, uintt h, oap::cuda::Kernel& kernel) {
  uint blocks[2];
  uint threads[2];
  uintt maxThreadsPerBlock = kernel.getMaxThreadsPerBlock();
  utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
  kernel.setBlocksCount(blocks[0], blocks[1]);
  kernel.setThreadsCount(threads[0], threads[1]);
}

bool execute(const char* functionName, math::Matrix* matrix, void** params,
                 uintt sharedMemory, oap::cuda::Kernel& kernel) {
  uintt w = oap::cuda::GetColumns(matrix);
  uintt h = oap::cuda::GetRows(matrix);
  prepareDims(w, h, kernel);
  kernel.setSharedMemory(sharedMemory);
  return ::oap::cuda::Kernel::Execute(functionName, params, kernel);
}

bool DEVICEKernel_DotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1,
                                 oap::cuda::Kernel& kernel) {
  void* params[] = {&output, &params0, &params1};
  return execute("CUDAKernel_DotProduct", output, params, 0, kernel);
}

bool DEVICEKernel_Transpose(math::Matrix* output, math::Matrix* params0,
                                oap::cuda::Kernel& kernel) {
  void* params[] = {&output, &params0};
  return execute("CUDAKernel_Transpose", output, params, 0, kernel);
}

bool DEVICEKernel_SetIdentity(math::Matrix* matrix,
                                  oap::cuda::Kernel& kernel) {
  void* params[] = {&matrix};
  return execute("CUDAKernel_SetIdentity", matrix, params, 0, kernel);
}

bool DEVICEKernel_Substract(math::Matrix* output, math::Matrix* params0,
                                math::Matrix* params1, oap::cuda::Kernel& kernel) {
  void* params[] = {&output, &params0, &params1};
  return execute("CUDAKernel_Substract", output, params, 0, kernel);
}

bool DEVICEKernel_CalcTriangularH(math::Matrix* H1, math::Matrix* Q,
                                      math::Matrix* R1, math::Matrix* Q1,
                                      math::Matrix* QJ, math::Matrix* Q2,
                                      math::Matrix* R2, math::Matrix* G,
                                      math::Matrix* GT, uintt columns,
                                      uintt rows, oap::cuda::Kernel& kernel) {
  void* params[] = {&H1, &Q, &R1, &Q1, &QJ, &Q2, &R2, &G, &GT};
  kernel.setDimensions(columns, rows);
  return oap::cuda::Kernel::Execute("CUDAKernel_CalculateTriangularH", params, kernel);
}
