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

#ifndef OAP_BUFFER_H
#define OAP_BUFFER_H

#include <algorithm>
#include <iterator>

#include <cstring>
#include <Math.h>

namespace utils
{

template<typename T>
class HostMemUtl
{
  public:
    T* alloc (uintt length)
    {
      return new T[length];
    }

    void free (T* buffer)
    {
      delete[] buffer;
    }

    void copyBuffer (T* dst, T* src, uintt length)
    {
      memcpy (dst, src, sizeof(T) * length);
    }

    template<typename Arg>
    void copy (T* dst, const Arg* src, uintt length)
    {
      memcpy (dst, src, sizeof(T) * length);
    }

    void copy (T* dst, const T* src, uintt length)
    {
      memcpy (dst, src, sizeof(T) * length);
    }
};

template <typename T, template<typename> class MemUtl>
class Buffer
{
  public:

    T* m_buffer;

    Buffer();
    virtual ~Buffer();

    Buffer (const Buffer&) = delete;
    Buffer (Buffer&&) = delete;
    Buffer& operator= (const Buffer&) = delete;
    Buffer& operator=  (Buffer&&) = delete;

    void realloc (uintt length);

    T* getBuffer ()
    {
      return m_buffer;
    }

    size_t GetSizeOfType() const
    { 
      return sizeof(T);
    }

    uintt getLength() const
    {
      return m_idx;
    }

    /*
    uintt push_back (const T& value)
    {
      uintt size = convertSize<T>();

      tryRealloc (size);
      m_memUtl.copy (&m_buffer[m_idx], &value, size);

      return increaseIdx (size);
    }

    uintt push_back (T&& value)
    {
      uintt size = convertSize<T>();

      tryRealloc (size);
      m_memUtl.copy (&m_buffer[m_idx], &value, size);

      return increaseIdx (size);
    }*/

    template<typename Arg>
    uintt push_back (const Arg& value)
    {
      uintt size = convertSize<Arg>();

      tryRealloc (size);

      if (std::is_same<Arg, T>::value)
      {
        m_memUtl.copy (&m_buffer[m_idx], &value, size);
      }
      else
      {
        m_memUtl.template copy<Arg> (&m_buffer[m_idx], &value, size);
      }

      return increaseIdx (size);
    }

    /*
    template<typename Arg>
    uintt push_back (Arg&& value)
    {
      uintt size = convertSize<Arg>();

      tryRealloc (size);

      if (std::is_same<Arg, T>::value)
      {
        m_memUtl.copy (&m_buffer[m_idx], &value, size);
      }
      else
      {
        m_memUtl.template copy<Arg> (&m_buffer[m_idx], &value, size);
      }

      return increaseIdx (size);
    }*/

    T get (uintt idx) const
    {
      checkIdx (idx);
      T value = m_buffer[idx];
      return value;
    }

    template<typename Arg>
    Arg get (uintt idx) const
    {
      checkIdx (idx);
      checkLength<Arg> (idx);

      Arg* ptr = reinterpret_cast<Arg*>(&m_buffer[idx]);
      Arg value = *ptr;

      return value;
    }

  protected:

    void tryRealloc (uintt value)
    {
      if (m_idx + value >= m_length)
      {
        realloc (m_idx + value);
      }
    }

    uintt increaseIdx (uintt value)
    {
      uintt idx = m_idx;
      m_idx += value;
      return idx;
    }

    void checkIdx (uintt idx) const
    {
      if (idx >= m_length)
      {
        throw std::runtime_error ("out of scope");
      }
    }

    template<typename Arg>
    void checkLength (uintt idx) const
    {
      uintt args = sizeof(Arg);
      uintt ts = sizeof(T);

      if (idx >= m_length)
      {
        throw std::runtime_error ("out of scope");
      }
    }

    template<typename Arg>
    uintt convertSize() const
    {
      uintt size = sizeof (Arg) / sizeof(T);
      if (sizeof (Arg) % sizeof(T) > 0)
      {
        size = size + 1;
      }
      return size;
    }

  private:

    uintt m_length;
    uintt m_idx;

    void free (T* buffer);
    T* alloc (uintt length);

    void allocBuffer (uintt length);
    void freeBuffer ();

    MemUtl<T> m_memUtl;
};

class ByteBuffer : public Buffer<char, HostMemUtl>
{
  public:
    ByteBuffer()
    {}

    virtual ~ByteBuffer()
    {}
};

template <typename T, template<typename> class MemUtl>
Buffer<T, MemUtl>::Buffer() : m_buffer(nullptr), m_length(0), m_idx(0)
{
}

template <typename T, template<typename> class MemUtl>
Buffer<T, MemUtl>::~Buffer()
{
  freeBuffer ();
}

template <typename T, template<typename> class MemUtl>
void Buffer<T, MemUtl>::realloc (uintt newLength)
{
  if (m_length == 0)
  {
    allocBuffer (newLength);
    return;
  }

  if (newLength > m_length)
  {
    T* buffer = m_buffer;
    uintt length = m_length;

    m_buffer = alloc (newLength);
    m_length = newLength;

    m_memUtl.copyBuffer (m_buffer, buffer, length);

    free (buffer);
  }
}

template <typename T, template<typename> class MemUtl>
void Buffer<T, MemUtl>::free (T* buffer)
{
  if (buffer != nullptr)
  {
    m_memUtl.free (buffer);
  }
}

template <typename T, template<typename> class MemUtl>
void Buffer<T, MemUtl>::allocBuffer (uintt length)
{
  if (m_buffer != nullptr)
  {
    freeBuffer ();
  }

  m_buffer = alloc(length);
  m_length = length;
}

template <typename T, template<typename> class MemUtl>
void Buffer<T, MemUtl>::freeBuffer ()
{
  free (m_buffer);
  m_length = 0;
  m_idx = 0;
}

template <typename T, template<typename> class MemUtl>
T* Buffer<T, MemUtl>::alloc (uintt length)
{
  return m_memUtl.alloc (length);
}

}

#endif
