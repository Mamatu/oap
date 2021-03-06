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

#ifndef OAP_KERNEL_CONFIG_H
#define OAP_KERNEL_CONFIG_H

#include <map>
#include <functional>
#include <sstream>

#include "Buffer.h"

#include "Math.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "IKernelExecutor.h"

#include "GenericCoreApi.h"
#include "GenericValidationApi.h"

#include "oapHostMatrixUtils.h"
#include "CuProcedures/CuKernelOperationsMacros.h"

#define CHECK_MATRIX(m) debugAssertMsg (m != NULL, "ComplexMatrix is nullptr.");

namespace oap
{

namespace generic
{

using SharedMemoryCallback = std::function<uintt(uintt blocks[2], uintt threads[2])>;

struct KernelConfig
{
  KernelConfig (bool _retrieveDims) : retrieveDims(_retrieveDims)
  {
    logAssert (retrieveDims);
  }

  KernelConfig (uintt _w, uintt _h) : w(_w), h(_h)
  {}

  bool retrieveDims = true;
  uintt w;
  uintt h;

  bool prepareDims = true;
  uint blocks[2];
  uint threads[2];
  uintt sharedMemorySize = 0;

  /**
   * \brief Shared memory callback which is invoke to define size of shared memory block (in bytes). 1st argument is blocks dims ([0] - x [1] - y), 2st argument of callback is threads dims ([0] - x [1] - y)
   *
   * The first argument of callback is blocks dims (as 2d array [0] - x [1] - y)
   * The second argument of callback is threads dims (as 2d array [0] - x [1] - y)
   * Type is @SharedMemoryCallback
   */
  SharedMemoryCallback smCallback = nullptr;
};

inline void prepareDims (oap::IKernelExecutor* kexec, uintt w, uintt h, uint blocks[2], uint threads[2])
{
  kexec->calculateThreadsBlocks (blocks, threads, w, h);
  kexec->setBlocksCount (blocks[0], blocks[1]);
  kexec->setThreadsCount (threads[0], threads[1]);
}

}
}

#endif
