#ifndef THREADSHOST_H
#define THREADSHOST_H

#include "Dim3.h"

class ThreadImpl;
class HostKernel;

class HostKernel {
 public:
  HostKernel();

  HostKernel(uintt columns, uintt rows);

  virtual ~HostKernel();

  void setDims(const dim3& gridDim, const dim3& blockDim);

  void calculateDims(uintt columns, uintt rows);

  void executeKernelAsync();

  void executeKernelSync();

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) = 0;

  enum ContextChange { CUDA_THREAD, CUDA_BLOCK };

  virtual void onChange(ContextChange contextChnage, const dim3& threadIdx,
                        const dim3& blockIdx) {}

  virtual void onSetDims(const dim3& gridDim, const dim3& blockDim) {}

  dim3 gridDim;
  dim3 blockDim;

 private:
  friend class ThreadImpl;
};

#endif  // THREADSHOST_H
