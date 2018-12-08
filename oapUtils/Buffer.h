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
#include <sstream>
#include <memory>
#include <Math.h>

namespace utils
{

template<typename T>
class HostMemUtl
{
  public:
    T* alloc (uintt length) const
    {
      return new T[length];
    }

    void free (T* buffer) const
    {
      delete[] buffer;
    }

    void copyBuffer (T* dst, T* src, uintt length) const
    {
      memcpy (dst, src, sizeof(T) * length);
    }

    template<typename Arg>
    void copy (T* dst, const Arg* src, uintt length) const
    {
      memcpy (dst, src, sizeof(T) * length);
    }

    void copyFromLoad (T* dst, const T* src, uintt length) const
    {
      memcpy (dst, src, sizeof (T) * length);
    }

    void copyToWrite (T* dst, const T* src, uintt length) const
    {
      memcpy (dst, src, sizeof(T) * length);
    }

    T get (T* buffer, uintt idx) const
    {
      return buffer [idx];
    }

    template<typename Arg>
    Arg get (T* buffer, uintt idx) const
    {
      T* valuePtr = &buffer[idx];
      Arg* arg = reinterpret_cast<Arg*>(valuePtr);
      return *arg;
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

    uintt getSizeOfBuffer() const
    {
      return sizeof (T) * m_length;
    }

    uintt getRealSizeOfBuffer() const
    {
      return sizeof (T) * m_realLength;
    }

    size_t getSizeOfType() const
    { 
      return sizeof(T);
    }

    uintt getLength() const
    {
      return m_length;
    }

    template<typename Arg>
    uintt push_back (const Arg& value)
    {
      uintt size = convertSize<Arg>();

      tryRealloc (size);

      if (std::is_same<Arg, T>::value)
      {
        m_memUtl.copy (&m_buffer[m_length], &value, size);
      }
      else
      {
        m_memUtl.template copy<Arg> (&m_buffer[m_length], &value, size);
      }

      return increaseIdx (size);
    }

    T get (uintt idx) const
    {
      checkIdx (idx);
      T value = m_memUtl.get (m_buffer, idx);
      return value;
    }

    template<typename Arg>
    Arg get (uintt idx) const
    {
      checkIdx (idx);
      checkLength<Arg> (idx);

      Arg value = m_memUtl.template get<Arg> (m_buffer, idx);

      return value;
    }

    void write (const std::string& path)
    {
      std::unique_ptr<FILE, decltype(&fclose)> f (fopen (path.c_str(), "wb"), fclose);

      if (!f)
      {
        std::stringstream sstr;
        sstr << "Cannot open " << path << " ";
        throw std::runtime_error (sstr.str());
      }

      size_t sizeOfT = getSizeOfType ();
      size_t length = getLength ();

      std::unique_ptr<T[]> hostBuffer (new T[length]);
      m_memUtl.copyToWrite (hostBuffer.get(), m_buffer, length);


      fwrite (&sizeOfT, sizeof (size_t), 1, f.get());
      fwrite (&length, sizeof(size_t), 1, f.get());
      fwrite (hostBuffer.get(), getSizeOfBuffer (), 1, f.get());
    }

    void read (const std::string& path)
    {
      std::unique_ptr<FILE, decltype(&fclose)> f (fopen (path.c_str(), "rb"), fclose);
    
      if (!f)
      {
        throw std::runtime_error ("File not found");
      }

      size_t sizeOfT = 0;
      size_t length = 0;

      fread (&sizeOfT, sizeof(size_t), 1, f.get());

      if (sizeOfT != getSizeOfType ())
      {
        throw std::runtime_error ("Loaded size of type is not equal with current class");
      }

      fread (&length, sizeof(size_t), 1, f.get());
  
      std::unique_ptr<T[]> hostBuffer (new T[length]);
      fread (hostBuffer.get (), getSizeOfBuffer (), 1, f.get());

      realloc (length);
      m_memUtl.copyFromLoad (m_buffer, hostBuffer.get (), length);
    }

  protected:

    void tryRealloc (uintt value)
    {
      if (m_length + value >= m_realLength)
      {
        realloc (m_length + value);
      }
    }

    uintt increaseIdx (uintt value)
    {
      uintt idx = m_length;
      m_length += value;
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

    uintt m_realLength; ///< Real number of allocated elements in buffer
    uintt m_length; ///< Number of used elements in buffer

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
Buffer<T, MemUtl>::Buffer() : m_buffer(nullptr), m_realLength(0), m_length(0)
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
  if (m_realLength == 0)
  {
    allocBuffer (newLength);
    return;
  }

  if (newLength > m_realLength)
  {
    T* buffer = m_buffer;
    uintt length = m_realLength;

    m_buffer = alloc (newLength);
    m_realLength = newLength;

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
  m_realLength = length;
}

template <typename T, template<typename> class MemUtl>
void Buffer<T, MemUtl>::freeBuffer ()
{
  free (m_buffer);
  m_realLength = 0;
  m_length = 0;
}

template <typename T, template<typename> class MemUtl>
T* Buffer<T, MemUtl>::alloc (uintt length)
{
  return m_memUtl.alloc (length);
}

}

#endif
