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

class OglaCudaStub;
class ThreadImpl;

class KernelStub {
 public:
  dim3 gridDim;
  dim3 blockDim;
  virtual ~KernelStub() {}

  void setDims(const dim3& _gridDim, const dim3& _blockDim) {
    blockDim = _blockDim;
    gridDim = _gridDim;
  }

  void calculateDims(uintt columns, uintt rows) {
    uintt blocks[2];
    uintt threads[2];
    utils::mapper::SetThreadsBlocks(blocks, threads, columns, rows, 1024);
    setDims(blocks, threads);
  }

 protected:
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) = 0;

  enum ContextChnage { CUDA_THREAD, CUDA_BLOCK };

  virtual void onChange(ContextChnage contextChnage, const dim3& threadIdx,
                        const dim3& blockIdx) {}

  friend class OglaCudaStub;
  friend class ThreadImpl;
};

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
  void executeKernelSync(KernelStub* cudaStub);

  void executeKernelAsync(KernelStub* cudaStub);
};

class ThreadImpl : public utils::Thread {
  dim3 m_threadIdx;
  dim3 m_blockIdx;
  KernelStub* m_cudaStub;
  std::vector<pthread_t>* m_pthreads;
  utils::sync::Barrier* m_barrier;
  utils::sync::Cond m_cond;
  utils::sync::Mutex m_mutex;
  bool m_cancontinue;

 public:
  ThreadImpl() : m_cudaStub(NULL), m_pthreads(NULL), m_barrier(NULL) {
    m_cancontinue = false;
  }

  void setKernelStub(KernelStub* cudaStub) { m_cudaStub = cudaStub; }

  void setThreadIdx(const dim3& threadIdx) { m_threadIdx = threadIdx; }

  void setBlockIdx(const dim3& blockIdx) { m_blockIdx = blockIdx; }

  void setPthreads(std::vector<pthread_t>* pthreads) { m_pthreads = pthreads; }

  void setBarrier(utils::sync::Barrier* barrier) { m_barrier = barrier; }

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
    const dim3 gridDim = threadImpl->m_cudaStub->gridDim;
    for (int fa = 0; fa < gridDim.x * gridDim.y; ++fa) {
      threadImpl->m_barrier->wait();
      ThreadIdx& ti = ThreadIdx::m_threadIdxs[pthread_self()];
      ti.setThreadIdx(threadImpl->m_threadIdx);
      ti.setBlockIdx(threadImpl->m_blockIdx);
      threadImpl->m_cudaStub->execute(threadImpl->m_threadIdx,
                                      threadImpl->m_blockIdx);
      threadImpl->m_cudaStub->onChange(KernelStub::CUDA_THREAD,
                                       threadImpl->m_threadIdx,
                                       threadImpl->m_blockIdx);

      threadImpl->m_mutex.lock();
      threadImpl->m_cancontinue = true;
      threadImpl->m_cond.signal();
      threadImpl->m_mutex.unlock();
    }
  }

  virtual void onRun(pthread_t threadId) {
    setFunction(ThreadImpl::Execute, this);
    ThreadIdx& ti = ThreadIdx::m_threadIdxs[threadId];
    ti.setBlockDim(m_cudaStub->blockDim);
    ti.setGridDim(m_cudaStub->gridDim);
    const dim3 blockDim = m_cudaStub->blockDim;
    m_pthreads->push_back(threadId);
    if (m_threadIdx.x == blockDim.x - 1 && m_threadIdx.y == blockDim.y - 1) {
      ThreadIdx::createBarrier(*m_pthreads);
    }
  }
};

inline void OglaCudaStub::executeKernelSync(KernelStub* cudaStub) {
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 gridDim = cudaStub->gridDim;
  const dim3 blockDim = cudaStub->blockDim;

  debugAssert(gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 &&
              blockDim.x != 0);

  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
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
          cudaStub->execute(threadIdx, blockIdx);
          cudaStub->onChange(KernelStub::CUDA_THREAD, threadIdx, blockIdx);
        }
      }
      cudaStub->onChange(KernelStub::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }
}

inline void OglaCudaStub::executeKernelAsync(KernelStub* cudaStub) {
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 blockDim = cudaStub->blockDim;
  const dim3 gridDim = cudaStub->gridDim;

  std::vector<ThreadImpl*> threads;
  std::vector<pthread_t> pthreads;
  unsigned int count = blockDim.y * blockDim.x + 1;
  utils::sync::Barrier barrier(count);
  // threads.resize(blockDim.x * blockDim.y);
  pthreads.reserve(blockDim.x * blockDim.y);

  debugAssert(gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 &&
              blockDim.x != 0);

  for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
    for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
      threadIdx.x = threadIdxX;
      threadIdx.y = threadIdxY;
      if (blockIdx.x == 0 && blockIdx.y == 0) {
        ThreadImpl* threadImpl = new ThreadImpl();
        threadImpl->setKernelStub(cudaStub);
        threadImpl->setThreadIdx(threadIdx);
        threadImpl->setBlockIdx(blockIdx);
        threadImpl->setPthreads(&pthreads);
        threadImpl->setBarrier(&barrier);
        threads.push_back(threadImpl);
      }
    }
  }

  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      blockIdx.x = blockIdxX;
      blockIdx.y = blockIdxY;
      for (size_t fa = 0; fa < threads.size(); ++fa) {
        threads.at(fa)->setBlockIdx(blockIdx);
        if (blockIdx.x == 0 && blockIdx.y == 0) {
          threads.at(fa)->run();
        }
      }
      barrier.wait();
      for (size_t fa = 0; fa < threads.size(); ++fa) {
        threads.at(fa)->waitOn();
      }
      cudaStub->onChange(KernelStub::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }

  for (size_t fa = 0; fa < threads.size(); ++fa) {
    threads.at(fa)->yield();
    delete threads.at(fa);
  }
  threads.clear();

  ThreadIdx::destroyBarrier(pthreads);
  pthreads.clear();
}
#endif /* OGLACUDAUTILS_H */
