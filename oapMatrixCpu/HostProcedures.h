/*
 * Copyright 2016, 2017 Marcin Matula
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

class HostProcedures {
 public:
  HostProcedures();
  virtual ~HostProcedures();

  void setThreadsCount(uintt threadsCount);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

  bool isEqual(math::Matrix* matrix1, math::Matrix* matrix2);

  void substract(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

  void dotProduct(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

  void transpose(math::Matrix* output, math::Matrix* matrix);

 private:
  uint m_threads[2];
  uint m_blocks[2];
  uint m_threadsCount;

  void prepare(math::Matrix* matrix, HostKernel& hostKernel);
};

#endif  // HOSTCOMPAREPROCEDURE_H
