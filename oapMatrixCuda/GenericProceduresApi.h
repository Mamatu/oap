/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef OAP_GENERIC_PROCEDURESAPI_H
#define OAP_GENERIC_PROCEDURESAPI_H

#include <map>
#include <sstream>

#include "Buffer.h"

#include "Math.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "IKernelExecutor.h"

namespace oap
{
namespace generic
{
  template<typename GetColumns, typename GetRows>
  class BasicMatrixDimApi
  {
    public:
      BasicMatrixDimApi (GetColumns&& _getColumns, GetRows&& _getRows) : getColumns (std::move (_getColumns)), getRows (std::move (_getRows))
      {}

      GetColumns&& getColumns;
      GetRows&& getRows;
  };

  template<typename GetMatrixInfo>
  class BasicMatrixApi
  {
    public:
      BasicMatrixApi (GetMatrixInfo&& _getMatrixInfo) : getMatrixInfo (std::move (_getMatrixInfo)) 
      {}

      GetMatrixInfo&& getMatrixInfo;
  };

  template<typename GetColumns, typename GetRows, typename GetMatrixInfo>
  class MatrixApi : BasicMatrixDimApi<GetColumns, GetRows>
  {
    public:
      MatrixApi (GetColumns&& _getColumns, GetRows&& _getRows, GetMatrixInfo&& _getMatrixInfo) :
        BasicMatrixDimApi<GetColumns, GetRows>(_getColumns, _getRows), getMatrixInfo(std::move (_getMatrixInfo))
      {}

      GetMatrixInfo&& getMatrixInfo;
  };


  template<typename GetMatrixInfo, typename Copy>
  class SumApi : public BasicMatrixApi<GetMatrixInfo>
  {
    public:
      SumApi(GetMatrixInfo&& _getMatrixInfo, Copy&& _copy):
             BasicMatrixApi<GetMatrixInfo>(_getMatrixInfo), copy (std::move(_copy))
      {}

      Copy&& copy;
  };

  template<typename HBuffer, typename DBuffer>
  class SumBuffers
  {
    public:

      SumBuffers (HBuffer& tbh1, DBuffer& tbd1, HBuffer& tbh2, DBuffer& tbd2):
              re (std::pair<HBuffer&, DBuffer&>(tbh1, tbd1)),
              im (std::pair<HBuffer&, DBuffer&>(tbh2, tbd2))
      {}

      std::pair<HBuffer&, DBuffer&> re;
      std::pair<HBuffer&, DBuffer&> im;
  };

  template<typename GetMatrixInfo, typename Copy, typename HBuffer, typename DBuffer>
  bool sum (floatt& reoutput, floatt& imoutput, math::Matrix* matrix, oap::IKernelExecutor* kexec, SumApi<GetMatrixInfo, Copy>& sumApi, SumBuffers<HBuffer, DBuffer>& sumBuffers)
  {
    auto minfo = sumApi.getMatrixInfo (matrix);

    const uintt w = minfo.columns ();
    const uintt h = minfo.rows ();

    uint blocks[2];
    uint threads[2];

    kexec->calculateThreadsBlocks(blocks, threads, w, h);
    kexec->setBlocksCount(blocks[0], blocks[1]);
    kexec->setThreadsCount(threads[0], threads[1]);

    const uintt length = blocks[0] * blocks[1];

    if (minfo.isRe)
    {
      sumBuffers.re.second.realloc (length);
      sumBuffers.re.first.realloc (length);
    }

    if (minfo.isIm)
    {
      sumBuffers.im.second.realloc (length);
      sumBuffers.im.first.realloc (length);
    }

    void* params[] = {&sumBuffers.re.second.m_buffer, &sumBuffers.im.second.m_buffer, &matrix};

    uint factor = 1;
    if (minfo.isIm && minfo.isRe)
    {
      factor = 2;
    }

    bool cuStatus = kexec->execute ("CUDAKernel_SumShared", params, factor * threads[0] * threads[1] * sizeof(floatt));

    reoutput = 0;
    imoutput = 0;

    auto sum = [length](const HBuffer& buffer) -> floatt
    {
      floatt sum = 0;
      for (size_t idx = 0; idx < length; ++idx)
      {
        sum += buffer.m_buffer[idx];
      }
      return sum;
    };

    if (minfo.isRe)
    {
      sumApi.copy (sumBuffers.re.first.m_buffer, sumBuffers.re.second.m_buffer, sumBuffers.re.first.getSizeOfBuffer());
      reoutput = sum (sumBuffers.re.first);
    }

    if (minfo.isIm)
    {
      sumApi.copy (sumBuffers.im.first.m_buffer, sumBuffers.im.second.m_buffer, sumBuffers.im.first.getSizeOfBuffer());
      imoutput = sum (sumBuffers.im.first);
    }

    return cuStatus;
  }

  template<typename GetMatrixInfo>
  bool crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, oap::IKernelExecutor* kexec, BasicMatrixApi<GetMatrixInfo>& api)
  {
    auto minfo = api.getMatrixInfo (output);

    if (minfo.isIm)
    {
      debugAssert ("CrossEntropy doesn't support complex matrix (only real)");
      return false;
    }

    const uintt w = minfo.columns ();
    const uintt h = minfo.rows ();

    uint blocks[2];
    uint threads[2];

    kexec->calculateThreadsBlocks(blocks, threads, w, h);
    kexec->setBlocksCount(blocks[0], blocks[1]);
    kexec->setThreadsCount(threads[0], threads[1]);   
      
    void* params[] = {&output, &params0, &params1};

    return kexec->execute ("CUDAKernel_CrossEntropy", params);
  }
}
}

#endif
