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

#ifndef OAP_THREADS_MAPPER_C_H
#define OAP_THREADS_MAPPER_C_H

#include <functional>
#include "Math.h"
#include "oapThreadsMapperS.h"

namespace oap
{

class ThreadsMapper
{
  public:
    using CreateCallback = std::function<oap::ThreadsMapperS* ()>;
    using DestroyCallback = std::function<void (oap::ThreadsMapperS*)>;

    ThreadsMapper (uintt width, uintt height, const CreateCallback& createCallback, const DestroyCallback& destroyCallback) :
      m_width(width), m_height(height), m_createCallback (createCallback), m_destroyCallback (destroyCallback)
    {}

    ThreadsMapper (uintt width, uintt height, CreateCallback&& createCallback, DestroyCallback&& destroyCallback) :
      m_width(width), m_height(height), m_createCallback (std::move(createCallback)), m_destroyCallback (std::move(destroyCallback))
    {}

    uintt getWidth () const
    {
      return m_width;
    }

    uintt getHeight () const
    {
      return m_height;
    }

    uintt getLength () const 
    {
      return getWidth() * getHeight();
    }

    oap::ThreadsMapperS* create () const
    {
      return m_createCallback ();
    }

    void destroy (oap::ThreadsMapperS* tms)
    {
      m_destroyCallback (tms);
    }

  private:
    uintt m_width;
    uintt m_height;
    CreateCallback m_createCallback;
    DestroyCallback m_destroyCallback;
};

}
#endif
