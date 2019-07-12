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
  class IKernelExecutor
  {
    public:
      IKernelExecutor();

      virtual ~IKernelExecutor();

      uint getThreadsX() const;
    
      uint getThreadsY() const;

      uint getThreadsZ() const;
    
      uint getBlocksX() const;
    
      uint getBlocksY() const;

      uint getBlocksZ() const;

      const uint* const getThreadsCount () const;

      const uint* const getBlocksCount () const;
    
      void setThreadsCount (intt x, intt y);
    
      void setBlocksCount (intt x, intt y);
    
      void setDimensions (uintt w, uintt h);
    
      void setSharedMemory (uintt sizeInBytes);

      uintt getSharedMemory () const
      {
        return m_sharedMemoryInBytes;
      }

      void setParams (void** params);
    
      int getParamsCount() const;
    
      void** getParams() const;

      virtual uint getMaxThreadsPerBlock() const = 0;

      bool execute (const char* functionName);
      bool execute (const char* functionName, void** params);
      bool execute (const char* functionName, void** params, uintt sharedMemorySize);

      virtual std::string getErrorMsg () const = 0;

      void calculateThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h);

      static void SetThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h, uint maxThreadsPerBlock);

    protected:
      virtual bool run (const char* function) = 0;

    private:
      uint m_threadsCount[3];
      uint m_blocksCount[3];

      uint m_sharedMemoryInBytes;
 
      void** m_params = nullptr;
      int m_paramsSize = 0;

      void reset ();
  };
}

#endif
