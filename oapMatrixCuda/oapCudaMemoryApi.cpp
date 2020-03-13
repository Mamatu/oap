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

#include <string.h>

#include "MatrixParser.h"
#include "ReferencesCounter.h"

#include "oapCudaMemoryApi.h"

#include "oapMemoryList.h"
#include "oapMemoryPrimitives.h"
#include "oapMemory_GenericApi.h"
#include "oapMemoryManager.h"

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

floatt* allocateBuffer (size_t length)
{
  floatt* buffer = static_cast<floatt*>(CudaUtils::AllocDeviceMem (length * sizeof (floatt)));
  g_memoryList.add (buffer, length);
  return buffer;
}

void deallocateBuffer (floatt* const buffer)
{
  g_memoryList.remove (buffer);
  CudaUtils::FreeDeviceMem (static_cast<const void*>(buffer));
}

oap::MemoryManagement<floatt*, decltype(allocateBuffer), decltype(deallocateBuffer), nullptr> g_memoryMng (allocateBuffer, deallocateBuffer);

floatt* allocateMem (const oap::MemoryDims& dims)
{
  floatt* raw = g_memoryMng.allocate (dims.width * dims.height);
  return raw;
}

void deallocateMem (const oap::Memory& memory)
{
  g_memoryMng.deallocate (memory.ptr);
}

}

oap::Memory NewMemory (const oap::MemoryDims& dims)
{
  return oap::generic::newMemory (dims, allocateMem); 
}

oap::Memory NewMemoryWithValues (const MemoryDims& dims, floatt value)
{
  return oap::generic::newMemoryWithValues (dims, value, [](const MemoryDims& dims, floatt value)
  {
    oap::Memory memory = NewMemory (dims);
    oap::Memory hmemory = oap::host::NewMemoryWithValues (dims, value);
    oap::cuda::CopyHostToDevice (memory, hmemory);
    return memory.ptr;
  });
}

oap::Memory NewMemoryDeviceCopy (const oap::Memory& src)
{
  return oap::generic::newMemoryCopy (src, [](floatt* const ptr, const oap::MemoryDims& dims)
  {
    oap::Memory memory = NewMemory (dims);
    oap::cuda::CopyDeviceToDevice (memory, {ptr, dims});
    return memory.ptr;
  });
}

oap::Memory NewMemoryHostCopy (const oap::Memory& src)
{
  return oap::generic::newMemoryCopy (src, [](floatt* const ptr, const oap::MemoryDims& dims)
  {
    oap::Memory memory = NewMemory (dims);
    oap::cuda::CopyHostToDevice (memory, {ptr, dims});
    return memory.ptr;
  });
}

oap::Memory NewMemoryDeviceCopyMem (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::newMemoryCopyMem (src, width, height, [](floatt* const ptr, const oap::MemoryDims& oldDims, const oap::MemoryDims& newDims)
  {
    oap::Memory memory = NewMemory (newDims);
    oap::cuda::CopyDeviceToDevice (memory, {ptr, oldDims});
    return memory.ptr;
  });
}

oap::Memory NewMemoryHostCopyMem (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::newMemoryCopyMem (src, width, height, [](floatt* const ptr, const oap::MemoryDims& oldDims, const oap::MemoryDims& newDims)
  {
    oap::Memory memory = NewMemory (newDims);
    oap::cuda::CopyHostToDevice (memory, {ptr, oldDims});
    return memory.ptr;
  });
}

oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::reuseMemory (src, width, height, [](floatt* ptr, const oap::MemoryDims& oldDims, const oap::MemoryDims& newDims)
  {
    return g_memoryMng.reuse (ptr);
  });
}

void DeleteMemory (const oap::Memory& mem)
{
  return oap::generic::deleteMemory (mem, [](const oap::Memory& mem)
  {
    deallocateMem (mem);
  });
}

oap::MemoryDims GetDims (const oap::Memory& mem)
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

}
}
