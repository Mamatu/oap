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

#ifndef OAP_GENERIC_PROCEDURES_NEW_API_H
#define OAP_GENERIC_PROCEDURES_NEW_API_H

#include "GenericProceduresApi.h"

namespace oap
{

namespace generic
{

inline bool dotProduct (oap::Memory& doutput, const oap::Memory& darg1, const oap::Memory& darg2, const oap::MemoryRegion_3_Args* dmatrices, oap::IKernelExecutor* kexec)
{
  uintt dims[2] = {doutput.dims.width, doutput.dims.height};

  const void* params[] = {&doutput.ptr, &doutput.dims, &darg1.ptr, &darg1.dims, &darg2.ptr, &darg2.dims, &dmatrices};
  const char* kname = "api2_CUDAKernel_DotProduct";

  oap::generic::Args args (doutput.dims.width, doutput.dims.height);
  args.retrieveDims = false;
  args.prepareDims = true;
  args.w = doutput.dims.width;
  args.h = doutput.dims.height;
  args.sharedMemorySize = args.w * args.h * sizeof(floatt);

  return executeKernel (kname, params, kexec, args);
}

}
}

#endif
