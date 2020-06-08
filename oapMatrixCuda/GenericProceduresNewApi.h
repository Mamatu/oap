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

#include "Matrix.h"
#include "oapThreadsMapperApi.h"
#include "GenericProceduresApi.h"

namespace oap
{
namespace generic
{

template<typename Matrices, typename CreateThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool addConstant (Matrices& output, const Matrices& params1, floatt dvalue, oap::IKernelExecutor* kexec, CreateThreadsMapper&& createThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());

  oap::threads::ThreadsMapper mapper = createThreadsMapper (output);

  uintt* buffer = static_cast<uintt*>(malloc (sizeof(uintt*) * mapper.getWidth() * mapper.getHeight()));
  mapper.map (buffer);

  const void* params[] = {&doutput, &dparams1, &dvalue, &buffer};
  const char* kname = "CUDAKernel_GenericApi_AddConstant";

  oap::generic::Args args (mapper.getWidth(), mapper.getHeight());
  args.retrieveDims = false;
  args.prepareDims = true;
  args.w = mapper.getWidth();
  args.h = mapper.getHeight();

  auto status = executeKernel (kname, params, kexec, args);  

  free (doutput);
  free (dparams1);
  free (buffer);

  return status;
}

template<typename GetThreadsCount>
bool dotProduct (oap::Memory& doutput, const oap::Memory& darg1, const oap::Memory& darg2, const oap::MemoryRegion_3_Args* dregions, uintt dregionsCount, oap::IKernelExecutor* kexec, GetThreadsCount&& getThreadsCount)
{
  auto threads = getThreadsCount (doutput);

  const void* params[] = {&doutput.ptr, &doutput.dims, &darg1.ptr, &darg1.dims, &darg2.ptr, &darg2.dims, &dregions, &dregionsCount};
  const char* kname = "api2_CUDAKernel_DotProduct";

  Args args (threads.first, threads.second);
  args.retrieveDims = false;
  args.prepareDims = true;
  args.w = threads.first;
  args.h = threads.second;
  args.sharedMemorySize = args.w * args.h * sizeof(floatt);

  return executeKernel (kname, params, kexec, args);
}

}
}

#endif
