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

#ifndef OAP_BUFFER_H
#define OAP_BUFFER_H

#include <algorithm>
#include <iterator>

#include <cstring>
#include <sstream>
#include <memory>

#include <Logger.h>
#include <Math.h>

namespace oap
{
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

    template<typename Arg>
    void set (T* buffer, uintt idx, const Arg* src, uintt length) const
    {
      memcpy (&buffer[idx], src, sizeof(T) * length);
    }

    template<typename Arg>
    void get (Arg* dst, uintt length, const T* buffer, uintt idx) const
    {
      memcpy (dst, &buffer[idx], sizeof(T) * length);
    }

    void copyBuffer (T* dst, const T* src, uintt length) const
    {
      memcpy (dst, src, sizeof(T) * length);
    }
};

template <typename T, template<typename> class MemUtl>
class Buffer
{
  public:

    T* m_buffer = nullptr;

    Buffer();

    /**
     * Loads binary content of file into this buffer
     */
    Buffer(const std::string& filepath);

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

    uintt getUsedSizeOfBuffer() const
    {
      return sizeof (T) * m_usedLength;
    }

    uintt getSizeOfBuffer() const
    {
      return sizeof (T) * m_length;
    }

    size_t getSizeOfType() const
    { 
      return sizeof(T);
    }

    uintt getUsedLength() const
    {
      return m_usedLength;
    }

    uintt getLength() const
    {
      return m_length;
    }

    template<typename Arg>
    uintt push_back (const Arg& value)
    {
      uintt argLen = getArgLength<Arg>();

      tryRealloc (argLen);

      m_memUtl.template set<Arg> (m_buffer, m_usedLength, &value, argLen);

      return increaseIdx (argLen);
    }

    template<typename Arg>
    uintt push_back (const Arg* buffer, size_t count)
    {
      uintt argLen = getArgLength<Arg>(count);

      tryRealloc (argLen);

      m_memUtl.template set<Arg> (m_buffer, m_usedLength, buffer, argLen);

      return increaseIdx (argLen);
    }

    T get (uintt idx) const
    {
      return this->template get<T> (idx);
    }

    template<typename Arg>
    Arg get (uintt idx) const
    {
      checkIdx (idx);
      checkLength<Arg> (idx);

      Arg value;
      m_memUtl.template get<Arg> (&value, getArgLength<Arg>(), m_buffer, idx);

      return value;
    }

    void get (T* buffer, size_t count, uintt idx) const
    {
      this->template get<T> (buffer, count, idx);
    }

    template<typename Arg>
    void get (Arg* buffer, size_t count, uintt idx) const
    {
      checkIdx (idx);
      checkLength<Arg> (idx, count);

      m_memUtl.template get<Arg> (buffer, getArgLength<Arg>(count), m_buffer, idx);
    }

    T read ()
    {
      return this->template read<T> ();
    }

    template<typename Arg>
    Arg read () const
    {
      Arg value = this->template get<Arg> (m_readIdx);
      increaseReadIdx<Arg> ();
      return value;
    }

    void read (T* buffer, size_t count) const
    {
      this->template read<T> (buffer, count);
    }

    template<typename Arg>
    void read (Arg* buffer, size_t count) const
    {
      this->template get<Arg> (buffer, count, m_readIdx);
      increaseReadIdx<Arg> (count);
    }

    void readReset ()
    {
      m_readIdx = 0;
    }

  public:
    void fwrite (const std::string& path) const
    {
      std::unique_ptr<FILE, decltype(&fclose)> f (fopen (path.c_str(), "wb"), fclose);

      if (!f)
      {
        std::stringstream sstream;
        sstream << "File \"" << path << "\" cannot be open to write";
        throw std::runtime_error (sstream.str());
      }

      size_t sizeOfT = getSizeOfType ();
      size_t length = getLength ();

      std::unique_ptr<T[]> hostBuffer (new T[length]);
      get (hostBuffer.get(), length, 0);

      auto std_fwrite = [&](const void* ptr, size_t size)
      {
        const size_t count = 1;
        size_t wcount = std::fwrite (ptr, size, count, f.get ());
        debugAssertMsg (wcount == count, "%s", strerror(errno));
      };

      std_fwrite (&sizeOfT, sizeof (size_t));
      std_fwrite (&length, sizeof (size_t));
      std_fwrite (hostBuffer.get (), getUsedSizeOfBuffer ());
    }

    void fread (const std::string& path)
    {
      std::unique_ptr<FILE, decltype(&fclose)> f (fopen (path.c_str(), "rb"), fclose);

      auto std_fread = [&](void* ptr, size_t size)
      {
        const size_t count = 1;
        size_t rcount = std::fread (ptr, size, count, f.get ());
        debugAssertMsg (rcount == count, "%s", strerror(errno));
      };

      if (!f)
      {
        std::stringstream sstream;
        sstream << "File \"" << path << "\" cannot be open to read.";
        throw std::runtime_error (sstream.str ());
      }

      size_t sizeOfT = 0;
      size_t length = 0;

      std_fread (&sizeOfT, sizeof(size_t));

      if (sizeOfT != getSizeOfType ())
      {
        throw std::runtime_error ("Loaded size of type is not equal with current class");
      }

      std_fread (&length, sizeof(size_t));

      std::unique_ptr<T[]> hostBuffer (new T[length]);
      std_fread (hostBuffer.get (), sizeOfT * length);

      push_back (hostBuffer.get (), length);
    }

  protected:

    void tryRealloc (uintt value)
    {
      if (m_usedLength + value >= m_length)
      {
        realloc (m_usedLength + value);
      }
    }

    uintt increaseIdx (uintt value)
    {
      uintt idx = m_usedLength;
      m_usedLength += value;
      return idx;
    }

    void checkIdx (uintt idx) const
    {
      if (idx >= m_usedLength)
      {
        std::stringstream stream;
        stream << "out of scope - idx too high idx: " << idx << ", m_usedLength: " << m_usedLength;
        throw std::runtime_error (stream.str ());
      }
    }

    template<typename Arg>
    void checkLength (uintt idx, uintt count = 1) const
    {
       uintt offset = getArgLength<Arg> (count);

      if (idx + offset > m_usedLength)
      {
        std::stringstream stream;
        stream << "out of scope - length too high idx: " << idx << "offset: " << offset << ", m_usedLength: " << m_usedLength;
        throw std::runtime_error (stream.str ());
      }
    }

    template<typename Arg>
    uintt getArgLength(size_t count = 1) const
    {
      uintt size = sizeof (Arg) / sizeof(T);
      if (sizeof (Arg) % sizeof(T) > 0)
      {
        size = size + 1;
      }
      return size * count;
    }

    template<typename Arg>
    uintt increaseReadIdx (size_t count = 1) const
    {
      m_readIdx += getArgLength<Arg> (count);
    }

  private:

    uintt m_length = 0; ///< Real number of allocated elements in buffer
    uintt m_usedLength = 0; ///< Number of used elements in buffer
    mutable uintt m_readIdx = 0; ///< Read index

    void free (T* buffer);
    T* alloc (uintt length);

    void allocBuffer (uintt length);
    void freeBuffer ();

    MemUtl<T> m_memUtl;
};

template <typename T, template<typename> class MemUtl>
Buffer<T, MemUtl>::Buffer()
{
}

template <typename T, template<typename> class MemUtl>
Buffer<T, MemUtl>::Buffer(const std::string& filepath)
{
  fread (filepath);
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

  m_buffer = alloc (length);
  m_length = length;
}

template <typename T, template<typename> class MemUtl>
void Buffer<T, MemUtl>::freeBuffer ()
{
  free (m_buffer);
  m_length = 0;
  m_usedLength = 0;
}

template <typename T, template<typename> class MemUtl>
T* Buffer<T, MemUtl>::alloc (uintt length)
{
  return m_memUtl.alloc (length);
}

}
}
#endif
