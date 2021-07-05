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

#include "oapHostKernelThread.hpp"
#include "oapHostKernelThread.hpp"

namespace oap
{

HostKernelThread::HostKernelThread() : Thread(), m_pthreads(NULL), m_barrier(NULL) {
  m_cancontinue = false;
}

void HostKernelThread::setExecCallback (ExecCallback execCallback) { m_execCallback = execCallback; }

void HostKernelThread::setBlockDim (const dim3& blockDim) { m_blockDim = blockDim; }

dim3 HostKernelThread::getThreadIdx () const { return m_threadIdx; }

dim3 HostKernelThread::getBlockIdx () const { return m_blockIdx; }

void HostKernelThread::setGridDim (const dim3& gridDim) { m_gridDim = gridDim; }

void HostKernelThread::setThreadIdx(const dim3& threadIdx) { m_threadIdx = threadIdx; }

void HostKernelThread::setBlockIdx(const dim3& blockIdx) { m_blockIdx = blockIdx; }

void HostKernelThread::setPthreads(std::vector<std::thread::id>* pthreads) { m_pthreads = pthreads; }

void HostKernelThread::setBarrier(oap::utils::sync::Barrier* barrier) { m_barrier = barrier; }

void HostKernelThread::setSharedBuffer (void* buffer) { m_sharedBuffer = buffer; }

HostKernelThread::~HostKernelThread()
{
  Thread::stop();
}

void HostKernelThread::waitOn()
{
  oap::utils::sync::MutexLocker locker(m_mutex);
  if (m_cancontinue == false)
  {
    m_cond.wait(m_mutex);
  }
  m_cancontinue = false;
}

void HostKernelThread::run()
{
  m_cancontinue = false;
  Thread::run (HostKernelThread::Execute, this);
}

std::thread::id HostKernelThread::get_id() const
{
  return Thread::get_id();
}

void HostKernelThread::Execute(void* ptr)
{
  HostKernelThread* threadImpl = static_cast<HostKernelThread*>(ptr);
  const int gridSize = threadImpl->m_gridDim.x * threadImpl->m_gridDim.y;
  for (int fa = 0; fa < gridSize; ++fa)
  {
    threadImpl->m_barrier->wait();

    ThreadIdx& ti = ThreadIdx::m_threadIdxs[std::this_thread::get_id()];
    ti.setThreadIdx(threadImpl->m_threadIdx);
    ti.setBlockIdx(threadImpl->m_blockIdx);

    //threadImpl->m_hostKernel->execute(threadImpl->m_threadIdx, threadImpl->m_blockIdx);
    //threadImpl->m_hostKernel->onChange(HostKernel::CUDA_THREAD, threadImpl->m_threadIdx, threadImpl->m_blockIdx);
    threadImpl->m_execCallback (threadImpl->m_threadIdx, threadImpl->m_blockIdx);

    {
      oap::utils::sync::MutexLocker locker(threadImpl->m_mutex);
      threadImpl->m_cancontinue = true;
    }
    threadImpl->m_cond.signal();
  }
}

void HostKernelThread::onRun(std::thread::id threadId)
{
  ThreadIdx& threadIndex = ThreadIdx::m_threadIdxs[threadId];
  threadIndex.setThreadIdx(m_threadIdx);
  threadIndex.setBlockIdx(m_blockIdx);
  threadIndex.setBlockDim(m_blockDim);
  threadIndex.setGridDim(m_gridDim);
  threadIndex.setSharedBuffer (m_sharedBuffer);
  m_pthreads->push_back (threadId);
  if (m_threadIdx.x == m_blockDim.x - 1 && m_threadIdx.y == m_blockDim.y - 1)
  {
    ThreadIdx::createBarrier(*m_pthreads);
  }
}
}
