/*
 * File:   dim3.h
 * Author: mmatula
 *
 * Created on March 10, 2015, 10:06 PM
 */

#ifndef DIM3_H
#define DIM3_H

#include <cuda.h>

#ifdef CUDATEST

#include <stdlib.h>
#include <map>
#include "Math.h"
#include "ThreadUtils.h"

class dim3 {
 public:
  dim3() {
    x = 0;
    y = 0;
    z = 1;
  }

  dim3(size_t tuple[2]) {
    x = tuple[0];
    y = tuple[1];
    z = 1;
  }

  dim3(uintt tuple[2]) {
    x = tuple[0];
    y = tuple[1];
    z = 1;
  }

  dim3(size_t x, size_t y) {
    this->x = x;
    this->y = y;
    z = 1;
  }

  void clear() {
    x = 0;
    y = 0;
  }

  size_t x;
  size_t y;
  size_t z;
};

typedef dim3 uint3;

extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;

void ResetCudaCtx();

class ThreadIdx {
  uint3 m_threadIdx;

  class BarrierMutex {
   public:
    utils::sync::Barrier m_barrier;
    utils::sync::Mutex m_mutex;
  };

  typedef std::map<pthread_t, BarrierMutex*> Barriers;
  static Barriers m_barriers;
  static utils::sync::Mutex m_barriersMutex;

 public:
  typedef std::map<pthread_t, ThreadIdx> ThreadIdxs;
  static ThreadIdxs m_threadIdxs;

  void setThreadIdx(const dim3& dim3);
  const uint3& getThreadIdx() const;
  static void createBarrier(const std::vector<pthread_t>& threads);
  static void destroyBarrier(const std::vector<pthread_t>& threads);
  static void wait();

  void clear();
};

#endif

#endif /* DIM3_H */
