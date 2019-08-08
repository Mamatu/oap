/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "Logger.h"

#include <map>
#include <functional>

std::map<std::string, std::function<void(void**)>> g_kernelsList =
{
  {"CUDAKernel_SumShared", HOSTKernel_SumSharedRaw},
  {"CUDAKernel_CrossEntropy", HOSTKernel_CrossEntropyRaw},
  {"CUDAKernel_Tanh", HOSTKernel_TanhRaw},
  {"CUDAKernel_Sigmoid", HOSTKernel_SigmoidRaw},
  {"CUDAKernel_DotProductDim", HOSTKernel_DotProductDimRaw},
  {"CUDAKernel_DotProduct", HOSTKernel_DotProductRaw},
  {"CUDAKernel_TensorProductDim", HOSTKernel_TensorProductDimRaw},
};

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

HostKernelExecutor::HostKernelExecutor(uint maxThreadsPerBlock) : m_maxThreadsPerBlock (maxThreadsPerBlock)
{}

HostKernelExecutor::~HostKernelExecutor()
{}

std::string HostKernelExecutor::getErrorMsg () const
{
  return "";
}

uint HostKernelExecutor::getMaxThreadsPerBlock() const
{
  return m_maxThreadsPerBlock;
}

bool HostKernelExecutor::run (const char* functionName)
{
  auto it = g_kernelsList.find (functionName);

  if (it == g_kernelsList.end ())
  {
    debugAssertMsg (false, "Function name is not registerd in g_kernelsList");
    return false;
  }

  HostKernelImpl hki (it->second, getParams());
  const oap::ExecutionParams& eParams = this->getExecutionParams ();
  dim3 blockDim (eParams.threadsCount);
  dim3 gridDim (eParams.blocksCount);
  hki.setDims (gridDim, blockDim);
  hki.setSharedMemory (eParams.sharedMemSize);

  hki.executeKernelAsync ();

  return true;
}

