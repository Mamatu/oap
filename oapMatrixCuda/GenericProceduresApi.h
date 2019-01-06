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

#include "CudaBuffer.h"
#include "HostBuffer.h"

#include "Math.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "CudaUtils.h"
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

  template<oap::Type type, oap::Type dtype>
  class SumBuffers
  {
    public:
      using TBufferH = oap::TBuffer<floatt, type>;
      using TBufferD = oap::TBuffer<floatt, dtype>;

      SumBuffers (TBufferH& tbh1, TBufferD& tbd1, TBufferH& tbh2, TBufferD& tbd2):
              re (std::pair<TBufferH&, TBufferD&>(tbh1, tbd1)),
              im (std::pair<TBufferH&, TBufferD&>(tbh2, tbd2))
      {}

      std::pair<TBufferH&, TBufferD&> re;
      std::pair<TBufferH&, TBufferD&> im;
  };

  template<typename GetMatrixInfo, typename Copy, oap::Type type, oap::Type dtype>
  bool sum (floatt& reoutput, floatt& imoutput, math::Matrix* matrix, oap::IKernelExecutor* kexec, SumApi<GetMatrixInfo, Copy>& sumApi, SumBuffers<type, dtype>& sumBuffers)
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

    auto sum = [length](const TBuffer<floatt, Type::HOST>& buffer) -> floatt
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
}
}

#endif
