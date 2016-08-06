/*
 * File:   oapCudaUtils.h
 * Author: mmatula
 *
 * Created on March 3, 2015, 11:15 PM
 */

#ifndef OGLACUDASTUB_H
#define OGLACUDASTUB_H

#include <vector>
#include "gtest/gtest.h"
#include "CuCore.h"
#include "Math.h"
#include "ThreadsMapper.h"
#include "ThreadUtils.h"
#include "HostKernel.h"

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
#endif /* OGLACUDAUTILS_H */
