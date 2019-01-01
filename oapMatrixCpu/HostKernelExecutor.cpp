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

#include "HostKernelExecutor.h"
#include "HostKernel.h"

#include "CudaKernelsList.h"

#include <map>
#include <functional>

std::map<std::string, std::function<void(void**)>> g_kernelsList = {{"HOSTKernel_SumSharedRaw", HOSTKernel_SumSharedRaw}};

class HostKernelImpl : public HostKernel
{
    std::function<void(void**)> m_function;
    void** m_params;
  public:
    HostKernelImpl (const std::function<void(void**)>& function, void** params) : m_function (function), m_params (params)
    {}

  protected:
    void execute(const dim3& threadIdx, const dim3& blockIdx)
    {
       m_function (m_params);
    }
};

HostKernelExecutor::HostKernelExecutor()
{}

HostKernelExecutor::~HostKernelExecutor()
{}

std::string HostKernelExecutor::getErrorMsg () const
{
  return "";
}

uint HostKernelExecutor::getMaxThreadsPerBlock() const
{
  return 32*1024;
}

bool HostKernelExecutor::run (const char* functionName)
{
  auto it = g_kernelsList.find (functionName);

  if (it == g_kernelsList.end ())
  {
    return false;
  }

  HostKernelImpl hki (it->second, getParams());
  dim3 blockDim (getThreadsCount());
  dim3 gridDim (getBlocksCount());
  hki.setDims (gridDim, blockDim);
  hki.setSharedMemory (getSharedMemory ());

  hki.executeKernelAsync ();

  return true;
}

