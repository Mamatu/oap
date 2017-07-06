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
