/*
 * File:   oglaCudaUtils.h
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

class OglaCudaStub : public testing::Test {
 public:
  OglaCudaStub() {}

  OglaCudaStub(const OglaCudaStub& orig) {}

  virtual ~OglaCudaStub() { ResetCudaCtx(); }

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

inline void OglaCudaStub::executeKernelSync(HostKernel* hostKernel) {
  hostKernel->executeKernelSync();
}

inline void OglaCudaStub::executeKernelAsync(HostKernel* hostKernel) {
  hostKernel->executeKernelAsync();
}
#endif /* OGLACUDAUTILS_H */
