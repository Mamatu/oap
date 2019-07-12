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

#include "HostKernel.h"
#include "ThreadUtils.h"
#include "ThreadsMapper.h"
#include "Dim3.h"

#include <memory>

class ThreadImpl;

HostKernel::HostKernel() {}

HostKernel::HostKernel(uintt columns, uintt rows) {
  calculateDims(columns, rows);
}

HostKernel::~HostKernel() {}

void HostKernel::setDims(const dim3& gridDim, const dim3& blockDim) {
  this->blockDim = blockDim;
  this->gridDim = gridDim;
  onSetDims(this->gridDim, this->blockDim);
}

void HostKernel::calculateDims(uintt columns, uintt rows) {
  uint blocks[2];
  uint threads[2];
  utils::mapper::SetThreadsBlocks(blocks, threads, columns, rows, 1024);
  setDims (dim3(blocks[0], blocks[1]), dim3(threads[0], threads[1]));
}

void HostKernel::setSharedMemory (size_t sizeInBytes)
{
  m_sharedMemorySize = sizeInBytes;
}

class ThreadImpl : public utils::Thread {
  dim3 m_threadIdx;
  dim3 m_blockIdx;
  HostKernel* m_hostKernel;
  std::vector<pthread_t>* m_pthreads;
  utils::sync::Barrier* m_barrier;
  utils::sync::Cond m_cond;
  utils::sync::Mutex m_mutex;
  bool m_cancontinue;
  void* m_sharedBuffer;

 public:
  ThreadImpl() : m_hostKernel(NULL), m_pthreads(NULL), m_barrier(NULL) {
    m_cancontinue = false;
  }

  void setHostKernel(HostKernel* threadFunction) {
    m_hostKernel = threadFunction;
  }

  void setThreadIdx(const dim3& threadIdx) { m_threadIdx = threadIdx; }

  void setBlockIdx(const dim3& blockIdx) { m_blockIdx = blockIdx; }

  void setPthreads(std::vector<pthread_t>* pthreads) { m_pthreads = pthreads; }

  void setBarrier(utils::sync::Barrier* barrier) { m_barrier = barrier; }
  
  void setSharedBuffer (void* buffer) { m_sharedBuffer = buffer; }

  virtual ~ThreadImpl() {}

  void waitOn() {
    m_mutex.lock();
    if (m_cancontinue == false) {
      m_cond.wait(m_mutex);
    }
    m_cancontinue = false;
    m_mutex.unlock();
  }

 protected:
  static void Execute(void* ptr) {
    ThreadImpl* threadImpl = static_cast<ThreadImpl*>(ptr);
    const int gridSize = threadImpl->m_hostKernel->gridDim.x * threadImpl->m_hostKernel->gridDim.y;
    for (int fa = 0; fa < gridSize; ++fa)
    {
      threadImpl->m_barrier->wait();

      ThreadIdx& ti = ThreadIdx::m_threadIdxs[pthread_self()];
      ti.setThreadIdx(threadImpl->m_threadIdx);
      ti.setBlockIdx(threadImpl->m_blockIdx);

      threadImpl->m_hostKernel->execute(threadImpl->m_threadIdx, threadImpl->m_blockIdx);
      threadImpl->m_hostKernel->onChange(HostKernel::CUDA_THREAD, threadImpl->m_threadIdx, threadImpl->m_blockIdx);

      threadImpl->m_mutex.lock();
      threadImpl->m_cancontinue = true;
      threadImpl->m_mutex.unlock();
      threadImpl->m_cond.signal();
    }
  }

  virtual void onRun(pthread_t threadId) {
    setFunction(ThreadImpl::Execute, this);
    ThreadIdx& threadIndex = ThreadIdx::m_threadIdxs[threadId];
    threadIndex.setThreadIdx(m_threadIdx);
    threadIndex.setBlockIdx(m_blockIdx);
    threadIndex.setBlockDim(m_hostKernel->blockDim);
    threadIndex.setGridDim(m_hostKernel->gridDim);
    threadIndex.setSharedBuffer (m_sharedBuffer);
    m_pthreads->push_back(threadId);
    if (m_threadIdx.x == m_hostKernel->blockDim.x - 1 &&
        m_threadIdx.y == m_hostKernel->blockDim.y - 1) {
      ThreadIdx::createBarrier(*m_pthreads);
    }
  }
};

void HostKernel::executeKernelAsync() {
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 blockDim = this->blockDim;
  const dim3 gridDim = this->gridDim;
  debugAssert(gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 &&
              blockDim.x != 0);

  std::vector<ThreadImpl*> threads;
  std::vector<pthread_t> pthreads;
  unsigned int count = blockDim.y * blockDim.x + 1;
  utils::sync::Barrier barrier(count);
  // threads.resize(blockDim.x * blockDim.y);
  pthreads.reserve (blockDim.x * blockDim.y);

  for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
    for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
      threadIdx.x = threadIdxX;
      threadIdx.y = threadIdxY;
      if (blockIdx.x == 0 && blockIdx.y == 0) {
        ThreadImpl* threadImpl = new ThreadImpl();
        threadImpl->setHostKernel(this);
        threadImpl->setThreadIdx(threadIdx);
        threadImpl->setBlockIdx(blockIdx);
        threadImpl->setPthreads(&pthreads);
        threadImpl->setBarrier(&barrier);
        threads.push_back(threadImpl);
      }
    }
  }

  std::unique_ptr<char[]> sharedMemory (nullptr);
  if (m_sharedMemorySize > 0)
  {
    sharedMemory.reset (new char[m_sharedMemorySize]);
  }
  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      blockIdx.x = blockIdxX;
      blockIdx.y = blockIdxY;

      for (size_t fa = 0; fa < threads.size(); ++fa) {
        threads.at(fa)->setBlockIdx (blockIdx);
        threads.at(fa)->setSharedBuffer (sharedMemory.get());
        if (blockIdx.x == 0 && blockIdx.y == 0) {
          threads.at(fa)->run();
        }
      }
      barrier.wait();
      for (size_t fa = 0; fa < threads.size(); ++fa) {
        threads.at(fa)->waitOn();
      }
      this->onChange(HostKernel::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }

  for (size_t fa = 0; fa < threads.size(); ++fa) {
    threads.at(fa)->join();
    delete threads.at(fa);
  }
  threads.clear();

  ThreadIdx::destroyBarrier(pthreads);
  pthreads.clear();
}

void HostKernel::executeKernelSync() {
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 gridDim = this->gridDim;
  const dim3 blockDim = this->blockDim;

  debugAssert(gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 &&
              blockDim.x != 0);

  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      std::unique_ptr<char[]> sharedMemory (nullptr);
      if (m_sharedMemorySize > 0)
      {
        sharedMemory.reset (new char[m_sharedMemorySize]);
      }
      for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
        for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
          threadIdx.x = threadIdxX;
          threadIdx.y = threadIdxY;
          blockIdx.x = blockIdxX;
          blockIdx.y = blockIdxY;
          ThreadIdx& ti = ThreadIdx::m_threadIdxs[pthread_self()];
          ti.setThreadIdx(threadIdx);
          ti.setBlockIdx(blockIdx);
          ti.setBlockDim(blockDim);
          ti.setGridDim(gridDim);
          ti.setSharedBuffer (sharedMemory.get ());
          this->execute(threadIdx, blockIdx);
          this->onChange(HostKernel::CUDA_THREAD, threadIdx, blockIdx);
        }
      }
      this->onChange(HostKernel::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }
}
