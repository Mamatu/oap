/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef HOSTPROCEDURE_H
#define HOSTPROCEDURE_H

#include "Matrix.h"
#include "HostKernel.h"
#include "HostKernelExecutor.h"

class HostProcedures {
 public:
  HostProcedures(uint maxThreadsPerBlock = 1024);
  virtual ~HostProcedures();

  void setThreadsCount(uintt threadsCount);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

  bool isEqual(math::Matrix* matrix1, math::Matrix* matrix2);

  void substract(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

  void dotProduct(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

  void transpose(math::Matrix* output, math::Matrix* matrix);

  void sum (floatt& reoutput, floatt& imoutput, math::Matrix* params0);

  void crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1);
 private:
  uint m_threads[2];
  uint m_blocks[2];
  uint m_threadsCount;

  HostKernelExecutor m_kernel;

  void prepare(math::Matrix* matrix, HostKernel& hostKernel);
};

#endif  // HOSTCOMPAREPROCEDURE_H
