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

#ifndef DIM3_H
#define DIM3_H

#ifdef OAP_CUDA_BUILD

#include <cuda.h>

#else

#include <stdlib.h>
#include <map>
#include <vector>

#include "Math.h"
#include "ThreadUtils.h"

class dim3 {
 public:
  dim3()
  {
    x = 0;
    y = 0;
    z = 1;
  }
  
  dim3 (const uint* const array)
  {
    x = array[0];
    y = array[1];
    z = array[2];
  }

  dim3 (uint x, uint y, uint z = 1)
  {
    set (x, y, z);
  }

  void clear()
  {
    x = 0;
    y = 0;
  }

  void set (uint _x, uint _y, uint _z = 1)
  {
    x = _x;
    y = _y;
    z = _z;
  }

  uint x;
  uint y;
  uint z;
};

typedef dim3 uint3;

void ResetCudaCtx();

class ThreadIdx {
  uint3 m_threadIdx;
  dim3 m_blockIdx;
  dim3 m_blockDim;
  dim3 m_gridDim;
  void* m_sharedBuffer;

  class BarrierMutex {
   public:
    oap::utils::sync::Barrier m_barrier;
    oap::utils::sync::Mutex m_mutex;
  };

  typedef std::map<pthread_t, BarrierMutex*> Barriers;
  static Barriers m_barriers;
  static oap::utils::sync::Mutex m_barriersMutex;

 public:
  ThreadIdx () : m_threadIdx (0, 0, 0), m_blockIdx (0, 0, 0), m_blockDim (0, 0, 1), m_gridDim (0, 0, 1)
  {}

  typedef std::map<pthread_t, ThreadIdx> ThreadIdxs;
  static ThreadIdxs m_threadIdxs;

  void setThreadIdx(const dim3& dim3);
  void setBlockIdx(const dim3& dim3);
  void setBlockDim(const dim3& dim3);
  void setGridDim(const dim3& dim3);
  void setSharedBuffer(void* buffer);

  const uint3& getThreadIdx() const;
  const dim3& getBlockIdx() const;
  const dim3& getBlockDim() const;
  const dim3& getGridDim() const;
  void* getSharedBuffer() const;

  static void createBarrier(const std::vector<pthread_t>& threads);
  static void destroyBarrier(const std::vector<pthread_t>& threads);
  static void wait();

  void clear();
};

#endif

#endif /* DIM3_H */
