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

#ifndef OAP_CUDA_MEMORY_API_H
#define OAP_CUDA_MEMORY_API_H

#include "ByteBuffer.h"
#include "Logger.h"

#include "MatrixPrinter.h"
#include "oapMemoryPrimitives.h"
#include "oapMemory_GenericApi.h"
#include "oapMemory_CommonApi.h"

#include "CudaUtils.h"

namespace oap
{
namespace cuda
{

oap::Memory NewMemory (const MemoryDim& dims);
oap::Memory NewMemoryWithValues (const MemoryDim& dims, floatt value);
oap::Memory NewMemoryDeviceCopy (const oap::Memory& src);
oap::Memory NewMemoryHostCopy (const oap::Memory& src);
oap::Memory NewMemoryDeviceCopyMem (const oap::Memory& src, uintt width, uintt height);
oap::Memory NewMemoryHostCopyMem (const oap::Memory& src, uintt width, uintt height);

oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height);
oap::Memory ReuseMemory (const oap::Memory& src);

void DeleteMemory (const oap::Memory& mem);

oap::MemoryDim GetDims (const oap::Memory& mem);
floatt* GetRawMemory (const oap::Memory& mem);

void CopyDeviceToDevice (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyDeviceToDevice (oap::Memory& dst, const oap::Memory& src);

void CopyHostToDevice (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostToDevice (oap::Memory& dst, const oap::Memory& src);

void CopyDeviceToHost (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyDeviceToHost (oap::Memory& dst, const oap::Memory& src);

void CopyDeviceToDeviceLinear (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyDeviceToDeviceLinear (oap::Memory& dst, const oap::Memory& src);

void CopyHostToDeviceLinear (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostToDeviceLinear (oap::Memory& dst, const oap::Memory& src);

void CopyDeviceToHostLinear (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyDeviceToHostLinear (oap::Memory& dst, const oap::Memory& src);

void CopyDeviceToHostBuffer (floatt* buffer, uintt length, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostBufferToDevice (oap::Memory& dst, const oap::MemoryRegion& dstReg, const floatt* buffer, uintt length);
void CopyDeviceBufferToDevice (oap::Memory& dst, const oap::MemoryRegion& dstReg, const floatt* buffer, uintt length);

uintt GetIdx (oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y);
floatt* GetPtr (oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y);
floatt GetValue (oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y);

uintt GetIdx (oap::Memory& memory, uintt x, uintt y);
floatt* GetPtr (oap::Memory& memory, uintt x, uintt y);
floatt GetValue (oap::Memory& memory, uintt x, uintt y);

template<typename MemoryVec>
oap::Memory NewMemoryBulkFromHost (const MemoryVec& vec, const oap::DataDirection& dd)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::cuda::NewMemory, CudaUtils::CopyHostToDevice);
}

template<typename MemoryVec, typename MemLocDimCallback>
oap::Memory NewMemoryBulkFromHost (const MemoryVec& vec, const oap::DataDirection& dd, MemLocDimCallback&& mldCallback)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::cuda::NewMemory, CudaUtils::CopyHostToDevice, mldCallback);
}

template<typename MemoryVec>
oap::Memory NewMemoryBulkFromDevice (const MemoryVec& vec, const oap::DataDirection& dd)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::cuda::NewMemory, CudaUtils::CopyDeviceToDevice);
}

template<typename MemoryVec, typename MemLocDimCallback>
oap::Memory NewMemoryBulkFromDevice (const MemoryVec& vec, const oap::DataDirection& dd, MemLocDimCallback&& mldCallback)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::cuda::NewMemory, CudaUtils::CopyDeviceToDevice, mldCallback);
}

}
}

#endif
