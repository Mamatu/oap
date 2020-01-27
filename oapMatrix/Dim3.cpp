/*
 * Copyright 2016 - 2019 Marcin Matula
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



#include "Dim3.h"

#ifndef OAP_CUDA_BUILD

void ResetCudaCtx() {
 // blockIdx.clear();
 // blockDim.clear();
  //gridDim.clear();
}

ThreadIdx::ThreadIdxs ThreadIdx::m_threadIdxs;

void ThreadIdx::clear() { m_threadIdx.clear(); }

void ThreadIdx::setThreadIdx(const dim3& dim3) { m_threadIdx = dim3; }

void ThreadIdx::setBlockIdx(const dim3& dim3) { m_blockIdx = dim3; }

void ThreadIdx::setBlockDim(const dim3& dim3) { m_blockDim = dim3; }

void ThreadIdx::setGridDim(const dim3& dim3) { m_gridDim = dim3; }

void ThreadIdx::setSharedBuffer(void* buffer)
{
  m_sharedBuffer = buffer;
}

const uint3& ThreadIdx::getThreadIdx() const { return m_threadIdx; }

const dim3& ThreadIdx::getBlockIdx() const { return m_blockIdx; }

const dim3& ThreadIdx::getBlockDim() const { return m_blockDim; }

const dim3& ThreadIdx::getGridDim() const { return m_gridDim; }

void* ThreadIdx::getSharedBuffer() const { return m_sharedBuffer; }

void ThreadIdx::createBarrier(const std::vector<pthread_t>& threads) {
  m_barriersMutex.lock();
  BarrierMutex* bm = NULL;
  for (size_t fa = 0; fa < threads.size(); ++fa) {
    if (fa == 0) {
      bm = new BarrierMutex();
      bm->m_barrier.init(threads.size());
    }
    m_barriers[threads[fa]] = bm;
  }
  m_barriersMutex.unlock();
}

void ThreadIdx::destroyBarrier(const std::vector<pthread_t>& threads) {
  m_barriersMutex.lock();
  for (size_t fa = 0; fa < threads.size(); ++fa) {
    if (fa == 0) {
      delete m_barriers[threads[fa]];
    }
    m_barriers.erase(threads[fa]);
  }
  m_barriersMutex.unlock();
}

void ThreadIdx::wait() {
  if (m_barriers.count(pthread_self()) > 0) {
    if (m_barriers[pthread_self()] != NULL) {
      m_barriers[pthread_self()]->m_barrier.wait();
    } else {
      debugAssert(m_barriers.count(pthread_self()) > 0);
    }
  } else {
    debugAssert(m_barriers[pthread_self()] != NULL);
  }
}

ThreadIdx::Barriers ThreadIdx::m_barriers;
oap::utils::sync::Mutex ThreadIdx::m_barriersMutex;

#endif
