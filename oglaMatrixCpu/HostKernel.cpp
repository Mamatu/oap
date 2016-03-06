#include "ThreadsHost.h"
#include "ThreadUtils.h"
#include "ThreadsMapper.h"
#include "Dim3.h"

class ThreadImpl;

void ThreadFunction::setDims(const dim3& gridDim, const dim3& blockDim) {
  m_blockDim = blockDim;
  m_gridDim = gridDim;
}

void ThreadFunction::calculateDims(uintt columns, uintt rows) {
  uintt blocks[2];
  uintt threads[2];
  utils::mapper::SetThreadsBlocks(blocks, threads, columns, rows, 1024);
  setDims(blocks, threads);
}

class ThreadImpl : public utils::Thread {
  dim3 m_threadIdx;
  dim3 m_blockIdx;
  ThreadFunction* m_threadFunction;
  std::vector<pthread_t>* m_pthreads;
  utils::sync::Barrier* m_barrier;
  utils::sync::Cond m_cond;
  utils::sync::Mutex m_mutex;
  bool m_cancontinue;

 public:
  ThreadImpl() : m_threadFunction(NULL), m_pthreads(NULL), m_barrier(NULL) {
    m_cancontinue = false;
  }

  void set(ThreadFunction* cudaStub, const dim3& threadIdx,
           const dim3& blockIdx, std::vector<pthread_t>* pthreads,
           utils::sync::Barrier* barrier) {
    m_threadFunction = cudaStub;
    m_threadIdx = threadIdx;
    m_blockIdx = blockIdx;
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
    for (int fa = 0; fa < threadImpl->m_threadFunction->m_gridDim.x *
                              threadImpl->m_threadFunction->m_gridDim.y;
         ++fa) {
      threadImpl->m_barrier->wait();
      threadImpl->m_threadFunction->execute(threadImpl->m_threadIdx,
                                            threadImpl->m_blockIdx);

      threadImpl->m_mutex.lock();
      threadImpl->m_cancontinue = true;
      threadImpl->m_cond.signal();
      threadImpl->m_mutex.unlock();
    }
  }

  virtual void onRun(pthread_t threadId) {
    setFunction(ThreadImpl::Execute, this);
    ThreadIdx& threadIndex = ThreadIdx::m_threadIdxs[threadId];
    threadIndex.setThreadIdx(m_threadIdx);
    threadIndex.setBlockIdx(m_blockIdx);
    threadIndex.setBlockDim(m_threadFunction->m_blockDim);
    threadIndex.setGridDim(m_threadFunction->m_gridDim);
    m_pthreads->push_back(threadId);
    if (m_threadIdx.x == m_threadFunction->m_blockDim.x - 1 &&
        m_threadIdx.y == m_threadFunction->m_blockDim.y - 1) {
      ThreadIdx::createBarrier(*m_pthreads);
    }
  }
};

void ThreadFunction::ExecuteKernelAsync(void (*Execute)(const dim3& threadIdx,
                                                     void* userData),
                                     void* userData) {
  class ThreadFunctionImpl : public ThreadFunction {};
}

void ThreadFunction::ExecuteKernelAsync(ThreadFunction* threadFunction) {
  uint3 threadIdx;
  dim3 blockIdx;

  const dim3& blockDim = threadFunction->m_blockDim;
  const dim3& gridDim = threadFunction->m_gridDim;

  std::vector<ThreadImpl*> threads;
  std::vector<pthread_t> pthreads;
  unsigned int count =
      threadFunction->m_blockDim.y * threadFunction->m_blockDim.x + 1;
  utils::sync::Barrier barrier(count);
  // threads.resize(blockDim.x * blockDim.y);
  pthreads.reserve(blockDim.x * blockDim.y);

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

          if (blockIdx.x == 0 && blockIdx.y == 0) {
            ThreadImpl* threadImpl = new ThreadImpl();
            threadImpl->set(threadFunction, threadIdx, blockIdx, &pthreads,
                            &barrier);
            threadImpl->run();
            threads.push_back(threadImpl);
          }
        }
      }
      barrier.wait();
      for (size_t fa = 0; fa < threads.size(); ++fa) {
        threads.at(fa)->waitOn();
      }
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
