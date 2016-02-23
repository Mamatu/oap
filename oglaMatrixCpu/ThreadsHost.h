#ifndef THREADSHOST_H
#define THREADSHOST_H

#include "Dim3.h"

class ThreadImpl;
class ThreadFunction;

class ThreadFunction {
 public:
  ThreadFunction();
  virtual ~ThreadFunction();

  void setDims(const dim3& gridDim, const dim3& blockDim);

  void calculateDims(uintt columns, uintt rows);

  static void ExecuteKernelAsync(ThreadFunction* threadFunction);

  static void ExecuteKernelAsync(void (*Execute)(const dim3& threadIdx,
                                                 void* userData),
                                 void* userData);

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) = 0;

 private:
  dim3 m_gridDim;
  dim3 m_blockDim;
  friend class ThreadImpl;
};

#endif  // THREADSHOST_H
