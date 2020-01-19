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

#ifndef OAP_HOST_MEMORY_API_H
#define OAP_HOST_MEMORY_API_H

#include "ByteBuffer.h"
#include "Logger.h"

#include "MatrixPrinter.h"
#include "oapMemoryPrimitives.h"
#include "oapGenericMemoryApi.h"

#define PRINT_MATRIX(m) logInfo ("%s %p\n%s %s", #m, m, oap::host::to_string(m).c_str(), oap::host::GetMatrixInfo(m).toString().c_str());
#define PRINT_DIMS_3_2(m) logInfo ("%s dims = {{%u, %u}, {%u, %u}, {%u, %u}} ", #m, m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]);
#define PRINT_DIMS_2_2_2(m) logInfo ("%s dims = {{{%u, %u}, {%u, %u}}, {{%u, %u}, {%u, %u}}} ", #m, m[0][0][0], m[0][0][1], m[0][1][0], m[0][1][1], m[1][0][0], m[1][0][1], m[1][1][0], m[1][1][1]);

namespace oap
{
namespace host
{

oap::Memory* NewMemory (const MemoryDims& dims);
oap::Memory* NewMemoryWithValues (const MemoryDims& dims, floatt value);

void DeleteMemory (const oap::Memory* mem);

oap::MemoryDims GetDims (const oap::Memory* mem);
floatt* getRawMemory (const oap::Memory* mem);


void CopyMemoryRegion (oap::Memory* dst, const oap::MemoryLoc& dstLoc, const oap::Memory* src, const oap::MemoryRegion& srcReg);

inline uintt GetIdx (oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return oap::utils::GetIdx (memory, reg, x, y);
}

inline floatt* GetPtr (oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return oap::utils::GetPtr (memory, reg, x, y);
}

inline floatt GetValue (oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return oap::utils::GetValue (memory, reg, x, y);
}

inline uintt GetIdx (oap::Memory* memory, uintt x, uintt y)
{
  return oap::utils::GetValue (memory, x, y);
}

inline floatt* GetPtr (oap::Memory* memory, uintt x, uintt y)
{
  return oap::utils::GetPtr (memory, x, y);
}

inline floatt GetValue (oap::Memory* memory, uintt x, uintt y)
{
  return oap::utils::GetValue (memory, x, y);
}

}
}

#endif
