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

#ifndef OAP_HOST_KERNEL_THREAD_H
#define OAP_HOST_KERNEL_THREAD_H

#include "ThreadUtils.hpp"
#include "ThreadsMapper.hpp"
#include "Dim3.hpp"

#include <memory>

namespace oap
{
class HostKernelThread : protected oap::utils::Thread
{
  dim3 m_threadIdx;
  dim3 m_blockIdx;
  std::vector<std::thread::id>* m_pthreads;
  oap::utils::sync::Barrier* m_barrier;
  oap::utils::sync::Cond m_cond;
  oap::utils::sync::Mutex m_mutex;
  bool m_cancontinue;
  void* m_sharedBuffer;

  dim3 m_gridDim;
  dim3 m_blockDim;

  using ExecCallback = std::function<void(dim3, dim3)>;
  ExecCallback m_execCallback;

  public:
    HostKernelThread();

    void setExecCallback (ExecCallback execCallback);

    void setBlockDim (const dim3& blockDim);

    void setGridDim (const dim3& gridDim);

    void setThreadIdx(const dim3& threadIdx);

    void setBlockIdx(const dim3& blockIdx);

    dim3 getThreadIdx () const;

    dim3 getBlockIdx () const;

    void setPthreads(std::vector<std::thread::id>* pthreads);

    void setBarrier(oap::utils::sync::Barrier* barrier);
  
    void setSharedBuffer (void* buffer);

    virtual ~HostKernelThread();

    void waitOn();

    void run();

    std::thread::id get_id() const;

    //void stop();
  protected:
    static void Execute(void* ptr);
    virtual void onRun(std::thread::id threadId);
};
}
#endif
