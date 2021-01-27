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

#ifndef OAP_GENERIC_PROCEDURES_NEW_API_H
#define OAP_GENERIC_PROCEDURES_NEW_API_H

#include "Matrix.h"
#include "oapThreadsMapperApi.h"
#include "oapThreadsMapperC.h"
#include "GenericProceduresApi.h"

namespace oap
{
namespace generic
{

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool addConstant (Matrices& output, const Matrices& params1, floatt dvalue, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  oapAssert (len == params1.size());
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = false;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dvalue, &tmS};
  const char* kname = "CUDAKernel_GenericApi_AddConst";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool add (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  oapAssert (len == params1.size());
  oapAssert (len == params2.size());

  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_Add";

  auto status = executeKernel (kname, params, kexec, args); 

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool subtract (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  oapAssert (len == params1.size());
  oapAssert (len == params2.size());

  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_Subtract";

  auto status = executeKernel (kname, params, kexec, args); 

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();
  args.sharedMemorySize = 0;

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_DotProduct";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dotProductShared (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();
  args.smCallback = [](uintt blocks[2], uintt threads[2])
  {
   
  };

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_DotProductShared";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  oapAssert (output.size() == params1.size());
  oapAssert (output.size() == params2.size());

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();
  args.sharedMemorySize = 0;

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_HadamardProduct";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy, typename GetMatrixInfo>
bool hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy, GetMatrixInfo&& getMatrixInfo)
{
  uintt len = output.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  oapAssert (output.size() == params1.size());
  oapAssert (output.size() == params2.size());

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
    oapAssert (getMatrixInfo(params2[idx]).columns() == 1);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();
  args.sharedMemorySize = 0;

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_PHadamardProduct";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams2 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams2, params2.data(), sizeof(math::Matrix*) * output.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();
  args.sharedMemorySize = 0;

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &dparams2, &tmS};
  const char* kname = "CUDAKernel_GenericApi_TensorProduct";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  free (dparams2);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool transpose (Matrices& output, const Matrices& params1, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = output.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {output[idx]};
    matrixArgs.push_back (line);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));
  math::Matrix** dparams1 = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * output.size()));

  memcpy (doutput, output.data(), sizeof(math::Matrix*) * output.size());
  memcpy (dparams1, params1.data(), sizeof(math::Matrix*) * output.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();
  args.sharedMemorySize = 0;

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* params[] = {&doutput, &dparams1, &tmS};
  const char* kname = "CUDAKernel_GenericApi_Transpose";

  auto status = executeKernel (kname, params, kexec, args);

  free (doutput);
  free (dparams1);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool func (const std::string& funcName, Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  uintt len = outputs.size();
  std::vector<std::vector<math::Matrix*>> matrixArgs;

  for (uintt idx = 0; idx < len; ++idx)
  {
    std::vector<math::Matrix*> line = {outputs[idx]};
    matrixArgs.push_back (line);
  }

  math::Matrix** doutput = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * outputs.size()));
  math::Matrix** dparams = static_cast<math::Matrix**>(malloc (sizeof(math::Matrix*) * outputs.size()));

  memcpy (doutput, outputs.data(), sizeof(math::Matrix*) * outputs.size());
  memcpy (dparams, params.data(), sizeof(math::Matrix*) * outputs.size());

  oap::ThreadsMapper mapper = getThreadsMapper (matrixArgs, oap::threads::ThreadsMapperAlgo::MATRIX_POS);

  oap::generic::Args args (mapper.getMinWidth(), mapper.getMinHeight());
  args.m_retrieveDims = false;
  args.m_prepareDims = true;
  args.w = mapper.getMinWidth();
  args.h = mapper.getMinHeight();

  args.prepareDims (kexec);
  oap::ThreadsMapperS* tmS = mapper.create (args.threads, args.blocks);

  const void* cu_params[] = {&doutput, &dparams, &tmS};
  const char* kname = funcName.c_str();

  auto status = executeKernel (kname, cu_params, kexec, args);

  free (doutput);
  free (dparams);
  mapper.destroy (tmS);

  return status;
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool sin (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Sin", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dsin (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DSin", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDSin (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDSin", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool tanh (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Tanh", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dtanh (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DTanh", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDTanh (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDTanh", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool sigmoid (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Sigmoid", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dsigmoid (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DSigmoid", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDSigmoid (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDSigmoid", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool prelu (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Prelu", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dprelu (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DPrelu", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDPrelu (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDPrelu", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool relu (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Relu", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool drelu (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DRelu", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDRelu (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDRelu", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool linear (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Linear", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dlinear (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DLinear", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDLinear (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDLinear", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool softplus (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_Softplus", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool dsoftplus (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_DSoftplus", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

template<typename Matrices, typename GetThreadsMapper, typename Malloc, typename Free, typename Memcpy>
bool multiplyDSoftplus (Matrices& outputs, const Matrices& params, oap::IKernelExecutor* kexec, GetThreadsMapper&& getThreadsMapper, Malloc&& malloc, Free&& free, Memcpy&& memcpy)
{
  return func ("CUDAKernel_GenericApi_MultiplyDSoftplus", outputs, params, kexec, getThreadsMapper, malloc, free, memcpy);
}

}
}

#endif
