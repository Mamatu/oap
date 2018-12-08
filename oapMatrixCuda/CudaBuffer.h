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

#ifndef OAP_CUDA_BUFFER_H
#define OAP_CUDA_BUFFER_H

#include "HostBuffer.h"
#include "CudaUtils.h"

namespace oap { namespace cuda {

template<typename T>
class HtoDMemUtl
{
  public:
    T* alloc (uintt length) const
    {
      void* ptr = CudaUtils::AllocDeviceMem (sizeof(T) * length);
      return static_cast<T*>(ptr);
    }

    void free (T* buffer) const
    {
      CudaUtils::FreeDeviceMem (buffer);
    }

    void copyBuffer (T* dst, const T* src, uintt length) const
    {
      CudaUtils::CopyDeviceToDevice (dst, src, sizeof(T) * length);
    }

    template<typename Arg>
    void copy (T* dst, const Arg* src, uintt length) const
    {
      CudaUtils::CopyHostToDevice (dst, src, sizeof(T) * length);
    }

    void copyFromLoad (T* dst, const T* src, uintt length) const
    {
      CudaUtils::CopyHostToDevice (dst, src, sizeof (T) * length);
    }

    void copyToWrite (T* dst, const T* src, uintt length) const
    {
      CudaUtils::CopyDeviceToHost (dst, src, sizeof (T) * length);
    }

    T get (T* buffer, uintt idx) const
    {
      T value;
      CudaUtils::CopyDeviceToHost (&value, &buffer[idx], sizeof(T));
      return value;
    }

    template<typename Arg>
    Arg get (T* buffer, uintt idx) const
    {
      Arg value;

      T* entry = buffer + idx;
      CudaUtils::CopyDeviceToHost (&value, reinterpret_cast<Arg*>(entry), sizeof(Arg));

      return value;
    }
};

template<typename T>
using HtoDBuffer = utils::Buffer<T, HtoDMemUtl>;

}

enum Type
{
  HOST, CUDA
};

template<typename T, Type type>
using TBuffer = typename std::conditional<type == Type::HOST, oap::host::HostBuffer<T>, cuda::HtoDBuffer<T>>::type;

}

#endif
