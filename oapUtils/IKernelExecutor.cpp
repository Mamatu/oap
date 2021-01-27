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

#include "IKernelExecutor.h"

#include "ThreadsMapper.h"

namespace oap
{
  IKernelExecutor::IKernelExecutor ()
  {
    reset ();
  }

  IKernelExecutor::~IKernelExecutor()
  {}

  void IKernelExecutor::setExecutionParams (const ExecutionParams& executionParams)
  {
    m_executionParams = executionParams;
  }

  const ExecutionParams& IKernelExecutor::getExecutionParams () const
  {
    return m_executionParams;
  }

  void IKernelExecutor::setBlocksCount (uint x, uint y)
  {
    m_executionParams.blocksCount [0] = x; 
    m_executionParams.blocksCount [1] = y; 
  }

  void IKernelExecutor::setThreadsCount (uint x, uint y)
  {
    m_executionParams.threadsCount [0] = x; 
    m_executionParams.threadsCount [1] = y; 
  }

  void IKernelExecutor::setSharedMemory (uint size)
  {
    m_executionParams.sharedMemSize = size;
  }
  
  void IKernelExecutor::setDimensions (uintt w, uintt h)
  {
    calculateThreadsBlocks (m_executionParams.blocksCount, m_executionParams.threadsCount, w, h);
  }

  void IKernelExecutor::setParams(const void** params) { m_params = params; }

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

  bool IKernelExecutor::execute (const char* functionName, const void** params)
  {
    setParams (params);
    return execute (functionName);
  }

  bool IKernelExecutor::execute (const char* functionName, const void** params, uintt sharedMemorySize)
  {
    m_executionParams.sharedMemSize = sharedMemorySize;
    return execute (functionName, params);
  }
  
  const void** IKernelExecutor::getParams() const
  {
    return m_params;
  }

  void IKernelExecutor::calculateThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h)
  {
    SetThreadsBlocks (blocks, threads, w, h, getMaxThreadsPerBlock());
  }

  void IKernelExecutor::SetThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h, uint maxThreadsPerBlock)
  {
    oap::utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
  }

  void IKernelExecutor::reset ()
  {
    m_executionParams.blocksCount[0] = 1;
    m_executionParams.blocksCount[1] = 1;
    m_executionParams.blocksCount[2] = 1;

    m_executionParams.threadsCount[0] = 1;
    m_executionParams.threadsCount[1] = 1;
    m_executionParams.threadsCount[2] = 1;

    m_executionParams.sharedMemSize = 0;

    m_params = nullptr;
    m_paramsSize = 0;
  }
}

