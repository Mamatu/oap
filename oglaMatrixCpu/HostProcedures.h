#ifndef HOSTPROCEDURE_H
#define HOSTPROCEDURE_H

#include "Matrix.h"
#include "HostKernel.h"

class HostProcedures {
 public:
  HostProcedures();
  virtual ~HostProcedures();

  void setThreadsCount(uintt threadsCount);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

  bool isEqual(math::Matrix* matrix1, math::Matrix* matrix2);

  void substract(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

 private:
  uintt m_threads[2];
  uintt m_blocks[2];
  uintt m_threadsCount;

  void prepare(math::Matrix* matrix, HostKernel& hostKernel);
};

#endif  // HOSTCOMPAREPROCEDURE_H
