/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "IKernelExecutor.h"

#include "ThreadsMapper.h"

namespace oap
{
  IKernelExecutor::IKernelExecutor()
  {
    reset ();
  }

  IKernelExecutor::~IKernelExecutor()
  {}

  uint IKernelExecutor::getThreadsX() const { return m_threadsCount[0]; }
  
  uint IKernelExecutor::getThreadsY() const { return m_threadsCount[1]; }

  uint IKernelExecutor::getThreadsZ() const { return m_threadsCount[2]; }
  
  uint IKernelExecutor::getBlocksX() const { return m_blocksCount[0]; }
  
  uint IKernelExecutor::getBlocksY() const { return m_blocksCount[1]; }

  uint IKernelExecutor::getBlocksZ() const { return m_blocksCount[2]; }
  
  const uint* const IKernelExecutor::getThreadsCount () const
  {
    return m_threadsCount;
  }

  const uint* const IKernelExecutor::getBlocksCount () const
  {
    return m_blocksCount;
  }
  
  void IKernelExecutor::setThreadsCount(intt x, intt y)
  {
    m_threadsCount[0] = x;
    m_threadsCount[1] = y;
  }
  
  void IKernelExecutor::setBlocksCount(intt x, intt y)
  {
    m_blocksCount[0] = x;
    m_blocksCount[1] = y;
  }
  
  void IKernelExecutor::setDimensions (uintt w, uintt h)
  {
    calculateThreadsBlocks (m_blocksCount, m_threadsCount, w, h);
  }

  void IKernelExecutor::setSharedMemory(uintt sizeInBytes)
  {
    m_sharedMemoryInBytes = sizeInBytes;
  }

  void IKernelExecutor::setParams(void** params) { m_params = params; }

  int IKernelExecutor::getParamsCount() const
  {
    return m_paramsSize;
  }

  bool IKernelExecutor::execute (const char* functionName)
  {
    bool status = run (functionName);
    reset();
    return status;
  }

  bool IKernelExecutor::execute (const char* functionName, void** params)
  {
    setParams (params);
    return execute (functionName);
  }

  bool IKernelExecutor::execute (const char* functionName, void** params, uintt sharedMemorySize)
  {
    setSharedMemory (sharedMemorySize);
    return execute (functionName, params);
  }
  
  void** IKernelExecutor::getParams() const
  {
    return m_params;
  }

  void IKernelExecutor::calculateThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h)
  {
    SetThreadsBlocks (blocks, threads, w, h, getMaxThreadsPerBlock());
  }

  void IKernelExecutor::SetThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h, uint maxThreadsPerBlock)
  {
    utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
  }

  void IKernelExecutor::reset ()
  {
    m_blocksCount[0] = 1;
    m_blocksCount[1] = 1;
    m_blocksCount[2] = 1;

    m_threadsCount[0] = 1;
    m_threadsCount[1] = 1;
    m_threadsCount[2] = 1;

    m_sharedMemoryInBytes = 0;

    m_params = nullptr;
    m_paramsSize = 0;
  }
}

