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

#ifndef OAP_IKERNEL_EXECUTOR_H
#define OAP_IKERNEL_EXECUTOR_H

#include <string>

#include "Math.h"
#include "Dim3.h"

namespace oap
{
  struct DeviceProperties
  {
    int maxThreadsCount[3];
    int maxBlocksCount[3];
  
    int maxThreadsPerBlock;
    int regsPerBlock;
    int sharedMemPerBlock;
  };

  struct ExecutionParams
  {
    uint threadsCount[3];
    uint blocksCount[3];
    uint sharedMemSize;
  };

  class IKernelExecutor
  {
    public:

      IKernelExecutor();
      virtual ~IKernelExecutor();

      void setExecutionParams (const ExecutionParams& executionParams);
      const ExecutionParams& getExecutionParams () const;

      void setBlocksCount (uint x, uint y);
      void setThreadsCount (uint x, uint y);
      void setSharedMemory (uint size);

      void setDimensions (uintt w, uintt h);

      void setParams (const void** params);
    
      int getParamsCount() const;
    
      const void** getParams() const;

      virtual uint getMaxThreadsPerBlock() const = 0;

      bool execute (const char* functionName);
      bool execute (const char* functionName, const void** params);
      bool execute (const char* functionName, const void** params, uintt sharedMemorySize);

      virtual std::string getErrorMsg () const = 0;

      void calculateThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h);

      static void SetThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h, uint maxThreadsPerBlock);

    protected:
      virtual bool run (const char* function) = 0;

    private:
      ExecutionParams m_executionParams;

      const void** m_params = nullptr;
      int m_paramsSize = 0;

      void reset ();
  };
}

#endif
