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

#include "HostKernel.h"
#include "ThreadUtils.h"
#include "ThreadsMapper.h"
#include "Dim3.h"

#include <memory>

HostKernel::HostKernel() {}

HostKernel::HostKernel(uintt columns, uintt rows) {
  calculateDims(columns, rows);
}

HostKernel::~HostKernel()
{}

void HostKernel::setDims(const dim3& gridDim, const dim3& blockDim) {
  this->blockDim = blockDim;
  this->gridDim = gridDim;
  onSetDims(this->gridDim, this->blockDim);
}

void HostKernel::calculateDims(uintt columns, uintt rows) {
  uint blocks[2];
  uint threads[2];
  oap::utils::mapper::SetThreadsBlocks(blocks, threads, columns, rows, 1024);
  setDims (dim3(blocks[0], blocks[1]), dim3(threads[0], threads[1]));
}

void HostKernel::setSharedMemory (size_t sizeInBytes)
{
  m_sharedMemorySize = sizeInBytes;
}

void HostKernel::executeKernelAsync()
{
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 blockDim = this->blockDim;
  const dim3 gridDim = this->gridDim;

  debugAssert(gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 && blockDim.x != 0);
  std::vector<oap::HostKernelThread*> m_threads;
  std::vector<std::thread::id> m_pthreads;

  unsigned int count = blockDim.y * blockDim.x + 1;

  oap::utils::sync::Barrier barrier (count);
  m_pthreads.reserve (blockDim.x * blockDim.y);

  for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY)
  {
    for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX)
    {
      threadIdx.x = threadIdxX;
      threadIdx.y = threadIdxY;
      if (blockIdx.x == 0 && blockIdx.y == 0)
      {
        oap::HostKernelThread* threadImpl = new oap::HostKernelThread();

        threadImpl->setExecCallback ([this](dim3 threadIdx, dim3 blockIdx)
            {
              execute(threadIdx, blockIdx);
              onChange(HostKernel::CUDA_THREAD, threadIdx, blockIdx);
            });
        threadImpl->setBlockDim (blockDim);
        threadImpl->setGridDim (gridDim);
        threadImpl->setThreadIdx(threadIdx);
        threadImpl->setBlockIdx(blockIdx);
        threadImpl->setPthreads(&m_pthreads);
        threadImpl->setBarrier(&barrier);
        m_threads.push_back(threadImpl);
      }
    }
  }

  std::unique_ptr<char[]> sharedMemory (nullptr);
  if (m_sharedMemorySize > 0)
  {
    sharedMemory.reset (new char[m_sharedMemorySize]);
  }
  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY)
  {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX)
    {
      blockIdx.x = blockIdxX;
      blockIdx.y = blockIdxY;

      for (size_t fa = 0; fa < m_threads.size(); ++fa)
      {
        m_threads.at(fa)->setBlockIdx (blockIdx);
        m_threads.at(fa)->setSharedBuffer (sharedMemory.get());
        if (blockIdx.x == 0 && blockIdx.y == 0)
        {
          m_threads.at(fa)->run();
        }
      }

      barrier.wait();

      for (size_t fa = 0; fa < m_threads.size(); ++fa)
      {
        m_threads.at(fa)->waitOn();
      }
      this->onChange(HostKernel::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }
  for (size_t fa = 0; fa < m_threads.size(); ++fa)
  {
    m_threads.at(fa)->stop();
    delete m_threads.at(fa);
  }
  m_threads.clear();

  ThreadIdx::destroyBarrier(m_pthreads);
  m_pthreads.clear();
}

void HostKernel::executeKernelSync()
{
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 gridDim = this->gridDim;
  const dim3 blockDim = this->blockDim;

  debugAssert (gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 && blockDim.x != 0);

  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY)
  {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX)
    {
      std::unique_ptr<char[]> sharedMemory (nullptr);
      if (m_sharedMemorySize > 0)
      {
        sharedMemory.reset (new char[m_sharedMemorySize]);
      }
      for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY)
      {
        for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX)
        {
          threadIdx.x = threadIdxX;
          threadIdx.y = threadIdxY;
          blockIdx.x = blockIdxX;
          blockIdx.y = blockIdxY;
          ThreadIdx& ti = ThreadIdx::m_threadIdxs[std::this_thread::get_id()];
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
