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

#ifndef OAP_THREADS_MAPPER_C_H
#define OAP_THREADS_MAPPER_C_H

#include <functional>
#include "Math.hpp"
#include "oapThreadsMapperS.hpp"

namespace oap
{

class ThreadsMapper
{
  public:
    using CreateCallback = std::function<oap::ThreadsMapperS* (uintt blockDim[2], uintt gridDim[2])>;
    using DestroyCallback = std::function<void (oap::ThreadsMapperS*)>;

    ThreadsMapper (uintt width, uintt height, const CreateCallback& createCallback, const DestroyCallback& destroyCallback) :
      m_minWidth(width), m_minHeight(height), m_createCallback (createCallback), m_destroyCallback (destroyCallback)
    {}

    ThreadsMapper (uintt width, uintt height, CreateCallback&& createCallback, DestroyCallback&& destroyCallback) :
      m_minWidth(width), m_minHeight(height), m_createCallback (std::move(createCallback)), m_destroyCallback (std::move(destroyCallback))
    {}

    uintt getMinWidth () const
    {
      return m_minWidth;
    }

    uintt getMinHeight () const
    {
      return m_minHeight;
    }

    uintt getMinLength () const
    {
      return getMinWidth() * getMinHeight();
    }

    oap::ThreadsMapperS* create (uintt blockDim[2], uintt gridDim[2]) const
    {
      return m_createCallback (blockDim, gridDim);
    }

    void destroy (oap::ThreadsMapperS* tms)
    {
      m_destroyCallback (tms);
    }

  private:
    uintt m_minWidth;
    uintt m_minHeight;
    CreateCallback m_createCallback;
    DestroyCallback m_destroyCallback;
};

}
#endif
