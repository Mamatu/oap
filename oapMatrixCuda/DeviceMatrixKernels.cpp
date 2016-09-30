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



#include "DeviceMatrixKernels.h"
#include "CudaUtils.h"
#include "ThreadsMapper.h"

void prepareDims(uintt w, uintt h, device::Kernel& kernel) {
  uintt blocks[2];
  uintt threads[2];
  uintt maxThreadsPerBlock = kernel.getMaxThreadsPerBlock();
  utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
  kernel.setBlocksCount(blocks[0], blocks[1]);
  kernel.setThreadsCount(threads[0], threads[1]);
}

CUresult execute(const char* functionName, math::Matrix* matrix, void** params,
                 uintt sharedMemory, device::Kernel& kernel) {
  uintt w = CudaUtils::GetColumns(matrix);
  uintt h = CudaUtils::GetRows(matrix);
  prepareDims(w, h, kernel);
  kernel.setSharedMemory(sharedMemory);
  return ::device::Kernel::Execute(functionName, params, kernel);
}

CUresult DEVICEKernel_DotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1,
                                 device::Kernel& kernel) {
  void* params[] = {&output, &params0, &params1};
  return execute("CUDAKernel_DotProduct", output, params, 0, kernel);
}

CUresult DEVICEKernel_Transpose(math::Matrix* output, math::Matrix* params0,
                                device::Kernel& kernel) {
  void* params[] = {&output, &params0};
  return execute("CUDAKernel_Transpose", output, params, 0, kernel);
}

CUresult DEVICEKernel_SetIdentity(math::Matrix* matrix,
                                  device::Kernel& kernel) {
  void* params[] = {&matrix};
  return execute("CUDAKernel_SetIdentity", matrix, params, 0, kernel);
}

CUresult DEVICEKernel_Substract(math::Matrix* output, math::Matrix* params0,
                                math::Matrix* params1, device::Kernel& kernel) {
  void* params[] = {&output, &params0, &params1};
  return execute("CUDAKernel_Substract", output, params, 0, kernel);
}

CUresult DEVICEKernel_CalcTriangularH(math::Matrix* H1, math::Matrix* Q,
                                      math::Matrix* R1, math::Matrix* Q1,
                                      math::Matrix* QJ, math::Matrix* Q2,
                                      math::Matrix* R2, math::Matrix* G,
                                      math::Matrix* GT, uintt columns,
                                      uintt rows, device::Kernel& kernel) {
  void* params[] = {&H1, &Q, &R1, &Q1, &QJ, &Q2, &R2, &G, &GT};
  kernel.setDimensions(columns, rows);
  return device::Kernel::Execute("CUDAKernel_CalculateTriangularH", params, kernel);
}