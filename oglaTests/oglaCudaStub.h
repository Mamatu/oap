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
  virtual void execute(const dim3& threadIdx) = 0;

  enum ContextChnage { CUDA_THREAD, CUDA_BLOCK };

  virtual void onChange(ContextChnage contextChnage, const dim3& threadIdx) {}

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

  void set(KernelStub* cudaStub, const dim3& threadIdx,
           std::vector<pthread_t>* pthreads, utils::sync::Barrier* barrier) {
    m_cudaStub = cudaStub;
    m_threadIdx = threadIdx;
    m_pthreads = pthreads;
    m_barrier = barrier;
  }

  virtual ~ThreadImpl() {}

  void setThreadIdx(const dim3& threadIdx) { m_threadIdx = threadIdx; }

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
    for (int fa = 0; fa < gridDim.x * gridDim.y; ++fa) {
      threadImpl->m_barrier->wait();
      threadImpl->m_cudaStub->execute(threadImpl->m_threadIdx);
      threadImpl->m_cudaStub->onChange(KernelStub::CUDA_THREAD,
                                       threadImpl->m_threadIdx);

      threadImpl->m_mutex.lock();
      threadImpl->m_cancontinue = true;
      threadImpl->m_cond.signal();
      threadImpl->m_mutex.unlock();
    }
  }

  virtual void onRun(pthread_t threadId) {
    setFunction(ThreadImpl::Execute, this);
    ThreadIdx::m_threadIdxs[threadId].setThreadIdx(m_threadIdx);
    m_pthreads->push_back(threadId);
    if (m_threadIdx.x == blockDim.x - 1 && m_threadIdx.y == blockDim.y - 1) {
      ThreadIdx::createBarrier(*m_pthreads);
    }
  }
};

inline void OglaCudaStub::executeKernelSync(KernelStub* cudaStub) {
  dim3 threadIdx;
  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
        for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
          threadIdx.x = threadIdxX;
          threadIdx.y = threadIdxY;
          blockIdx.x = blockIdxX;
          blockIdx.y = blockIdxY;
          ThreadIdx::m_threadIdxs[pthread_self()].setThreadIdx(threadIdx);
          cudaStub->execute(threadIdx);
          cudaStub->onChange(KernelStub::CUDA_THREAD, threadIdx);
        }
      }
      cudaStub->onChange(KernelStub::CUDA_BLOCK, threadIdx);
    }
  }
}

inline void OglaCudaStub::executeKernelAsync(KernelStub* cudaStub) {
  dim3 threadIdx;

  std::vector<ThreadImpl*> threads;
  std::vector<pthread_t> pthreads;
  unsigned int count = blockDim.y * blockDim.x + 1;
  utils::sync::Barrier barrier(count);
  //threads.resize(blockDim.x * blockDim.y);
  pthreads.reserve(blockDim.x * blockDim.y);

  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
        for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
          threadIdx.x = threadIdxX;
          threadIdx.y = threadIdxY;
          blockIdx.x = blockIdxX;
          blockIdx.y = blockIdxY;

          if (blockIdx.x == 0 && blockIdx.y == 0) {
            ThreadImpl* threadImpl = new ThreadImpl();
            threadImpl->set(cudaStub, threadIdx, &pthreads, &barrier);
            threadImpl->run();
            threads.push_back(threadImpl);
          }
        }
      }
      barrier.wait();
      for (size_t fa = 0; fa < threads.size(); ++fa) {
        threads.at(fa)->waitOn();
      }
      cudaStub->onChange(KernelStub::CUDA_BLOCK, threadIdx);
    }
  }
  for (size_t fa = 0; fa < threads.size(); ++fa) {
    threads.at(fa)->yield();
  }
  threads.clear();

  ThreadIdx::destroyBarrier(pthreads);
  pthreads.clear();
}
#endif /* OGLACUDAUTILS_H */
