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
  {"CUDAKernel_SumShared", proxy_HOSTKernel_SumShared},
  {"CUDAKernel_CrossEntropy", proxy_HOSTKernel_CrossEntropy},
  {"CUDAKernel_DotProductDim", proxy_HOSTKernel_DotProductDim},
  {"CUDAKernel_DotProductPeriodic", proxy_HOSTKernel_DotProductPeriodic},
  {"CUDAKernel_DotProductDimPeriodic", proxy_HOSTKernel_DotProductDimPeriodic},
  {"CUDAKernel_DotProduct", proxy_HOSTKernel_specific_DotProduct},
  {"CUDAKernel_DotProductShared", proxy_HOSTKernel_DotProductShared},
  {"CUDAKernel_TensorProductDim", proxy_HOSTKernel_TensorProductDim},
  {"CUDAKernel_Tanh", proxy_HOSTKernel_Tanh},
  {"CUDAKernel_Sigmoid", proxy_HOSTKernel_Sigmoid},
  {"CUDAKernel_Sin", proxy_HOSTKernel_Sin},
  {"CUDAKernel_TanhDim", proxy_HOSTKernel_TanhDim},
  {"CUDAKernel_SigmoidDim", proxy_HOSTKernel_SigmoidDim},
  {"CUDAKernel_SinDim", proxy_HOSTKernel_SinDim},
  {"CUDAKernel_TanhDimPeriodic", proxy_HOSTKernel_TanhDimPeriodic},
  {"CUDAKernel_SigmoidDimPeriodic", proxy_HOSTKernel_SigmoidDimPeriodic},
  {"CUDAKernel_SinDimPeriodic", proxy_HOSTKernel_SinDimPeriodic},
  {"CUDAKernel_QRHT", proxy_HOSTKernel_QRHT},
  {"CUDAKernel_SetIdentity", proxy_HOSTKernel_SetIdentity},
  {"CUDAKernel_SetVector", proxy_HOSTKernel_setVector},
  {"CUDAKernel_GetVector", proxy_HOSTKernel_getVector},

  {"CUDAKernel_PRelu", proxy_HOSTKernel_PRelu},
  {"CUDAKernel_DPRelu", proxy_HOSTKernel_DPRelu},
  {"CUDAKernel_PReluDim", proxy_HOSTKernel_PReluDim},
  {"CUDAKernel_DPReluDim", proxy_HOSTKernel_DPReluDim},
  {"CUDAKernel_PReluDimPeriodic", proxy_HOSTKernel_PReluDimPeriodic},
  {"CUDAKernel_DPReluDimPeriodic", proxy_HOSTKernel_DPReluDimPeriodic},

  {"CUDAKernel_Relu", proxy_HOSTKernel_Relu},
  {"CUDAKernel_DRelu", proxy_HOSTKernel_DRelu},
  {"CUDAKernel_ReluDim", proxy_HOSTKernel_ReluDim},
  {"CUDAKernel_DReluDim", proxy_HOSTKernel_DReluDim},
  {"CUDAKernel_ReluDimPeriodic", proxy_HOSTKernel_ReluDimPeriodic},
  {"CUDAKernel_DReluDimPeriodic", proxy_HOSTKernel_DReluDimPeriodic}
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

void HostKernelExecutor::setMaxThreadsPerBlock (uintt threadsPerBlock)
{
  m_maxThreadsPerBlock = threadsPerBlock;
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

