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

class OglaCudaStub : public testing::Test {
 public:
  OglaCudaStub() {}

  OglaCudaStub(const OglaCudaStub& orig) {}

  virtual ~OglaCudaStub() { ResetCudaCtx(); }

  virtual void SetUp() {}

  virtual void TearDown() {}

  class KernelStub {
   public:
    virtual ~KernelStub() {}

    void setDims(const Dim3& _gridDim, const Dim3& _blockDim) {
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
    virtual void execute(const Dim3& threadIdx) = 0;

    enum ContextChnage { CUDA_THREAD, CUDA_BLOCK };

    virtual void onChange(ContextChnage contextChnage, const Dim3& threadIdx) {}

    friend class OglaCudaStub;
  };

  /**
   * Kernel is executed sequently in one thread.
   * This can be used to kernel/device functions which doesn't
   * synchronization procedures.
   * .
   * @param cudaStub
   */
  void executeKernelSync(KernelStub* cudaStub) {
    Dim3 threadIdx;
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
            cudaStub->onChange(OglaCudaStub::KernelStub::CUDA_THREAD,
                               threadIdx);
          }
        }
        cudaStub->onChange(OglaCudaStub::KernelStub::CUDA_BLOCK, threadIdx);
      }
    }
  }

  void executeKernelAsync(KernelStub* cudaStub) {
    class ThreadImpl : public utils::Thread {
      Dim3 m_threadIdx;
      KernelStub* m_cudaStub;

     public:
      ThreadImpl(KernelStub* cudaStub, const Dim3& threadIdx)
          : m_cudaStub(cudaStub), m_threadIdx(threadIdx) {}

      virtual ~ThreadImpl() {}

     protected:
      static void Execute(void* ptr) {
        ThreadImpl* threadImpl = static_cast<ThreadImpl*>(ptr);
        threadImpl->m_cudaStub->execute(threadImpl->m_threadIdx);
        threadImpl->m_cudaStub->onChange(OglaCudaStub::KernelStub::CUDA_THREAD,
                                         threadImpl->m_threadIdx);
      }

      virtual void onRun(pthread_t threadId) {
        setFunction(ThreadImpl::Execute, this);
        ThreadIdx::m_threadIdxs[threadId].setThreadIdx(m_threadIdx);
      }
    };
    Dim3 threadIdx;

    std::vector<utils::Thread*> threads;

    for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
      for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
        for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
          for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
            threadIdx.x = threadIdxX;
            threadIdx.y = threadIdxY;
            blockIdx.x = blockIdxX;
            blockIdx.y = blockIdxY;

            ThreadImpl* thread = new ThreadImpl(cudaStub, threadIdx);
            threads.push_back(thread);
            thread->run();
          }
        }
        for (size_t fa = 0; fa < threads.size(); ++fa) {
          threads[fa]->yield();
          delete threads[fa];
        }
        threads.clear();
        cudaStub->onChange(OglaCudaStub::KernelStub::CUDA_BLOCK, threadIdx);
      }
    }
  }
};

#endif /* OGLACUDAUTILS_H */
