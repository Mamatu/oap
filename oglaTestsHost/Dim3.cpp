#include "Dim3.h"

#ifdef CUDATEST

dim3 blockIdx;
dim3 blockDim;
dim3 gridDim;

void ResetCudaCtx() {
  blockIdx.clear();
  blockDim.clear();
  gridDim.clear();
}

ThreadIdx::ThreadIdxs ThreadIdx::m_threadIdxs;

void ThreadIdx::clear() { m_threadIdx.clear(); }

void ThreadIdx::setThreadIdx(const dim3& dim3) { m_threadIdx = dim3; }

const uint3& ThreadIdx::getThreadIdx() const { return m_threadIdx; }

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
utils::sync::Mutex ThreadIdx::m_barriersMutex;

#endif
