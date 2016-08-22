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



#include <math.h>
#include "TestProcedures.h"
#include "CudaUtils.h"

CuTest::CuTest() : m_cuResult(CUDA_SUCCESS) {
  const char* pathes[3];
  pathes[0] =
      "/home/mmatula/Oap/oapCudaTests/dist/Debug/GNU-Linux-x86/"
      "liboapCudaTests.cubin";
  pathes[1] =
      "/home/mmatula/Oap/oapCudaTests/dist/Debug/albert/"
      "liboapCudaTests.cubin";
  pathes[2] = NULL;
  m_kernel.load(pathes);
}

bool CuTest::test1() {
  size_t length = m_kernel.getMaxThreadsPerBlock() / 8;
  int* buffer1 =
      static_cast<int*>(CudaUtils::AllocDeviceMem(sizeof(int) * length));
  int* doutput = CudaUtils::AllocDeviceObj<int>();
  size_t offset = sqrt(length);
  m_kernel.setDimensions(sqrt(length), sqrt(length));
  uint blocksCount = m_kernel.getBlocksX() * m_kernel.getBlocksY();
  int* bin =
      static_cast<int*>(CudaUtils::AllocDeviceMem(sizeof(int) * blocksCount));
  int* bout =
      static_cast<int*>(CudaUtils::AllocDeviceMem(sizeof(int) * blocksCount));
  void* params[] = {&doutput, &buffer1, &offset, &length, &bin, &bout};
  m_kernel.setParams(params);
  m_cuResult = m_kernel.execute("CUDAKernel_Test1");
  int output = 0;
  CudaUtils::CopyDeviceToHost(&output, doutput, sizeof(int));
  CudaUtils::FreeDeviceObj(doutput);
  CudaUtils::FreeDeviceMem(bin);
  CudaUtils::FreeDeviceMem(bout);
  return output == length;
}

bool CuTest::test2() {
  size_t length = m_kernel.getMaxThreadsPerBlock() / 8;
  int* buffer1 =
      static_cast<int*>(CudaUtils::AllocDeviceMem(sizeof(int) * length));
  int* doutput = CudaUtils::AllocDeviceObj<int>();
  size_t offset = sqrt(length);
  m_kernel.setDimensions(sqrt(length), sqrt(length));
  uint blocksCount = m_kernel.getBlocksX() * m_kernel.getBlocksY();
  int* mutex = static_cast<int*>(CudaUtils::AllocDeviceObj<int>(0));
  void* params[] = {&doutput, &buffer1, &offset, &length, &mutex};
  m_kernel.setParams(params);
  m_cuResult = m_kernel.execute("CUDAKernel_Test2");
  int output = 0;
  CudaUtils::CopyDeviceToHost(&output, doutput, sizeof(int));
  CudaUtils::FreeDeviceObj(doutput);
  CudaUtils::FreeDeviceMem(mutex);
  return output == length;
}

CUresult CuTest::getStatus() const { return m_cuResult; }
