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




#ifndef OAPCUDASTUB_H
#define OAPCUDASTUB_H

#include <vector>
#include "gtest/gtest.h"
#include "CuCore.hpp"
#include "Math.hpp"
#include "ThreadsMapper.hpp"
#include "ThreadUtils.hpp"
#include "HostKernel.hpp"

class OapCudaStub : public testing::Test {
 public:
  OapCudaStub() {}

  OapCudaStub(const OapCudaStub& orig) {}

  virtual ~OapCudaStub() { ResetCudaCtx(); }

  virtual void SetUp() {}

  virtual void TearDown() {}

  /**
   * Kernel is executed sequently in one thread.
   * This can be used to kernel/device functions which doesn't
   * synchronization procedures.
   * .
   * @param cudaStub
   */
  void executeKernelSync(HostKernel* hostKernel);

  void executeKernelAsync(HostKernel* hostKernel);
};

inline void OapCudaStub::executeKernelSync(HostKernel* hostKernel) {
  hostKernel->executeKernelSync();
}

inline void OapCudaStub::executeKernelAsync(HostKernel* hostKernel) {
  hostKernel->executeKernelAsync();
}
#endif /* OAPCUDAUTILS_H */
