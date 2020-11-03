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

#include <string.h>

#include "MatrixParser.h"
#include "ReferencesCounter.h"

#include "oapCudaMemoryApi.h"

#include "oapMemoryList.h"
#include "oapMemoryPrimitives.h"
#include "oapMemory_GenericApi.h"
#include "oapMemoryCounter.h"

#include "CudaUtils.h"

#define ReIsNotNULL(m) m->reValues != nullptr
#define ImIsNotNULL(m) m->imValues != nullptr

#ifdef DEBUG
/*
std::ostream& operator<<(std::ostream& output, const math::Matrix*& matrix)
{
  return output << matrix << ", [" << matrix->columns << ", " << matrix->rows
         << "]";
}
*/
#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#else

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillWithValue (floatt* values, floatt value, uintt length)
{
  math::Memset (values, value, length);
}

inline void fillRePart(math::Matrix* output, floatt value)
{
  fillWithValue (output->re.ptr, value, output->re.dims.width * output->re.dims.height);
}

inline void fillImPart(math::Matrix* output, floatt value)
{
  fillWithValue (output->im.ptr, value, output->im.dims.width * output->im.dims.height);
}

namespace oap
{
namespace cuda
{

namespace
{

MemoryList g_memoryList ("MEMORY_CUDA");
MemoryCounter g_memoryCounter;

floatt* allocateMem (const oap::MemoryDim& dims)
{
  const uintt length = dims.width * dims.height;
  floatt* buffer = static_cast<floatt*>(CudaUtils::AllocDeviceMem (length * sizeof (floatt)));
  g_memoryList.add (buffer, length);
  return buffer;
}

void deallocateMem (const oap::Memory& memory)
{
  CudaUtils::FreeDeviceMem (static_cast<const void*>(memory.ptr));
  g_memoryList.remove (memory.ptr);
}

}

oap::Memory NewMemory (const oap::MemoryDim& dims)
{
  return oap::generic::newMemory (dims, allocateMem, [](floatt* ptr)
    {
      g_memoryCounter.increase (ptr);
    });
}

oap::Memory NewMemoryWithValues (const MemoryDim& dims, floatt value)
{
  oap::Memory memory = NewMemory (dims);
  oap::Memory hmemory = oap::host::NewMemoryWithValues (dims, value);
  oap::cuda::CopyHostToDevice (memory, hmemory);
  oap::host::DeleteMemory (hmemory);
  return memory;
}

oap::Memory NewMemoryDeviceCopy (const oap::Memory& src)
{
  oap::Memory memory = NewMemory (src.dims);
  oap::generic::copyMemory (memory, src, CudaUtils::CopyDeviceToDevice);
  return memory;
}

oap::Memory NewMemoryHostCopy (const oap::Memory& src)
{
  oap::Memory memory = NewMemory (src.dims);
  oap::generic::copyMemory (memory, src, CudaUtils::CopyHostToDevice);
  return memory;
}

oap::Memory NewMemoryDeviceCopyMem (const oap::Memory& src, uintt width, uintt height)
{
  oap::Memory memory = NewMemory ({width, height});
  oap::generic::copyMemory (memory, src, CudaUtils::CopyDeviceToDevice);
  return memory;
}

oap::Memory NewMemoryHostCopyMem (const oap::Memory& src, uintt width, uintt height)
{
  oap::Memory memory = NewMemory ({width, height});
  oap::generic::copyMemory (memory, src, CudaUtils::CopyHostToDevice);
  return memory;
}

oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::reuseMemory (src, width, height, [](floatt* ptr)
      {
        g_memoryCounter.increase (ptr);
      });
}

oap::Memory ReuseMemory (const oap::Memory& src)
{
  return oap::generic::reuseMemory (src, src.dims.width, src.dims.height, [](floatt* ptr)
      {
        g_memoryCounter.increase (ptr);
      });
}

void DeleteMemory (const oap::Memory& mem)
{
  oap::generic::deleteMemory (mem, deallocateMem, [](floatt* ptr) -> uintt
      {
        return g_memoryCounter.decrease (ptr);
      });
}

oap::MemoryDim GetDims (const oap::Memory& mem)
{
  return mem.dims;
}

floatt* GetRawMemory (const oap::Memory& mem)
{
  return mem.ptr;
}

void CopyDeviceToDevice (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copy (dstPtr, dstDims, dstLoc, srcPtr, srcDims, srcReg, CudaUtils::CopyDeviceToDevice);
}

void CopyDeviceToDevice (oap::Memory& dst, const oap::Memory& src)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copy (dstPtr, dstDims, {0, 0}, srcPtr, srcDims, {{0, 0}, srcDims}, CudaUtils::CopyDeviceToDevice);
}

void CopyHostToDevice (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = src.dims;
  auto* srcPtr = src.ptr;

  oap::generic::copy (dstPtr, dstDims, dstLoc, srcPtr, srcDims, srcReg, CudaUtils::CopyHostToDevice);
}

void CopyHostToDevice (oap::Memory& dst, const oap::Memory& src)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = src.dims;
  auto* srcPtr = src.ptr;

  oap::generic::copy (dstPtr, dstDims, {0, 0}, srcPtr, srcDims, {{0, 0}, srcDims}, CudaUtils::CopyHostToDevice);
}

void CopyDeviceToHost (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  const auto& dstDims = dst.dims;
  auto* dstPtr = dst.ptr;

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copy (dstPtr, dstDims, dstLoc, srcPtr, srcDims, srcReg, CudaUtils::CopyDeviceToHost);
}

void CopyDeviceToHost (oap::Memory& dst, const oap::Memory& src)
{
  const auto& dstDims = dst.dims;
  auto* dstPtr = dst.ptr;

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copy (dstPtr, dstDims, {0, 0}, srcPtr, srcDims, {{0, 0}, srcDims}, CudaUtils::CopyDeviceToHost);
}

void CopyDeviceToDeviceLinear (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copyLinear (dstPtr, dstDims, dstLoc, srcPtr, srcDims, srcReg, CudaUtils::CopyDeviceToDevice);
}

void CopyDeviceToDeviceLinear (oap::Memory& dst, const oap::Memory& src)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copyLinear (dstPtr, dstDims, {0, 0}, srcPtr, srcDims, {{0, 0}, srcDims}, CudaUtils::CopyDeviceToDevice);
}

void CopyHostToDeviceLinear (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = src.dims;
  auto* srcPtr = src.ptr;

  oap::generic::copyLinear (dstPtr, dstDims, dstLoc, srcPtr, srcDims, srcReg, CudaUtils::CopyHostToDevice);
}

void CopyHostToDeviceLinear (oap::Memory& dst, const oap::Memory& src)
{
  const auto& dstDims = oap::cuda::GetDims (dst);
  auto* dstPtr = oap::cuda::GetRawMemory (dst);

  const auto& srcDims = src.dims;
  auto* srcPtr = src.ptr;

  oap::generic::copyLinear (dstPtr, dstDims, {0, 0}, srcPtr, srcDims, {{0, 0}, srcDims}, CudaUtils::CopyHostToDevice);
}

void CopyDeviceToHostLinear (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  const auto& dstDims = dst.dims;
  auto* dstPtr = dst.ptr;

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copyLinear (dstPtr, dstDims, dstLoc, srcPtr, srcDims, srcReg, CudaUtils::CopyDeviceToHost);
}

void CopyDeviceToHostLinear (oap::Memory& dst, const oap::Memory& src)
{
  const auto& dstDims = dst.dims;
  auto* dstPtr = dst.ptr;

  const auto& srcDims = oap::cuda::GetDims (src);
  auto* srcPtr = oap::cuda::GetRawMemory (src);

  oap::generic::copyLinear (dstPtr, dstDims, {0, 0}, srcPtr, srcDims, {{0, 0}, srcDims}, CudaUtils::CopyDeviceToHost);
}

void CopyDeviceToHostBuffer (floatt* buffer, uintt length, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  oap::generic::copyMemoryRegionToBuffer (buffer, length, src.ptr, src.dims, srcReg, CudaUtils::CopyDeviceToHost);
}

void CopyHostBufferToDevice (oap::Memory& dst, const oap::MemoryRegion& dstReg, const floatt* buffer, uintt length)
{
  oap::generic::copyBufferToMemoryRegion (dst.ptr, dst.dims, dstReg, buffer, length, CudaUtils::CopyDeviceToHost);
}

void CopyDeviceBufferToDevice (oap::Memory& dst, const oap::MemoryRegion& dstReg, const floatt* buffer, uintt length)
{
  oap::generic::copyBufferToMemoryRegion (dst.ptr, dst.dims, dstReg, buffer, length, CudaUtils::CopyDeviceToDevice);
}
}
}
