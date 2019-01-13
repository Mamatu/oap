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

#ifndef OAP_SMARTPOINTERUTILS_H
#define OAP_SMARTPOINTERUTILS_H

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <vector>

#include "Matrix.h"

namespace deleters
{

using MatrixDeleter = std::function<void(const math::Matrix*)>;

class MatricesDeleter
{
    size_t m_count;
    MatrixDeleter m_deleter;
  
  public:
    MatricesDeleter (size_t count, deleters::MatrixDeleter deleter) : 
      m_count(count), m_deleter(deleter) {}

    MatricesDeleter& operator() (math::Matrix** matrices)
    {
      for (size_t idx = 0; idx < m_count; ++idx)
      {
        m_deleter (matrices[idx]);
      }
      delete[] matrices;
      return *this;
    }

    void setCount (size_t count)
    {
      m_count = count;
    }
};

class SharedPtrGetDeleter
{
  public:

    template<typename Deleter, typename StdMatrixPtr>
    Deleter& get_deleter (StdMatrixPtr& stdMatrixPtr)
    {
      return *std::get_deleter<Deleter, typename StdMatrixPtr::element_type> (stdMatrixPtr);
    }
};

class UniquePtrGetDeleter
{
  public:

    template<typename Deleter, typename StdMatrixPtr>
    Deleter& get_deleter (StdMatrixPtr& stdMatrixPtr)
    {
      return stdMatrixPtr.get_deleter ();
    }
};

template<typename Deleter, typename SharedPtrType, typename UniquePtrType, typename StdMatrixPtr>
Deleter& get_deleter (StdMatrixPtr& stdMatrixPtr)
{
  constexpr bool isSharedPtr = std::is_same<SharedPtrType, StdMatrixPtr>::value;
  constexpr bool isUniquePtr = std::is_same<UniquePtrType, StdMatrixPtr>::value;
  
  static_assert ((isSharedPtr && !isUniquePtr) || (!isSharedPtr && isUniquePtr), "StdMatrixPtr is unsupported type");
 
  typename std::conditional<isSharedPtr, SharedPtrGetDeleter, UniquePtrGetDeleter>:: type obj;
  
  return obj.template get_deleter<Deleter, StdMatrixPtr> (stdMatrixPtr);
}
}

namespace reset
{
template<typename StdMatrixPtr, typename Deleter>
class SharedPtrReset
{
    StdMatrixPtr& m_stdMatrixPtr;
    Deleter&& m_deleter;

  public:
    SharedPtrReset (StdMatrixPtr& stdMatrixPtr, Deleter&& deleter) : m_stdMatrixPtr (stdMatrixPtr), m_deleter (deleter)
    {}

    void reset (typename StdMatrixPtr::element_type* t)
    {
      m_stdMatrixPtr.reset (t, m_deleter);
    }
};

template<typename StdMatrixPtr, typename Deleter>
class UniquePtrReset
{
    StdMatrixPtr& m_stdMatrixPtr;
    Deleter m_deleter;

  public:
    UniquePtrReset (StdMatrixPtr& stdMatrixPtr, Deleter&& deleter) : m_stdMatrixPtr (stdMatrixPtr), m_deleter (deleter)
    {}

    void reset (typename StdMatrixPtr::element_type* t)
    {
      m_stdMatrixPtr.reset (t);
    }
};

template<typename SharedPtrType, typename UniquePtrType, typename StdMatrixPtr, typename Deleter>
void reset (StdMatrixPtr& stdMatrixPtr, Deleter&& deleter, typename StdMatrixPtr::element_type* t)
{
  constexpr bool isSharedPtr = std::is_same<SharedPtrType, StdMatrixPtr>::value;
  constexpr bool isUniquePtr = std::is_same<UniquePtrType, StdMatrixPtr>::value;
  
  static_assert ((isSharedPtr && !isUniquePtr) || (!isSharedPtr && isUniquePtr), "StdMatrixPtr is unsupported type");
 
  typename std::conditional<isSharedPtr, reset::SharedPtrReset<StdMatrixPtr, decltype(deleter)>, reset::UniquePtrReset<StdMatrixPtr, decltype(deleter)>>:: type obj(stdMatrixPtr, deleter);
  
  obj.reset (t);
}
}

namespace smartptr_utils
{

  template<typename T>
  T* makeArray(size_t count)
  {
    T* array = new T[count];
    memset(array, 0, sizeof(T) * count);
    return array;
  }

  template<template<typename, typename> class Container, typename T, class Allocator>
  T* makeArray(const Container<T, Allocator>& vec)
  {
    T* array = new T [vec.size()];
    std::copy (vec.begin(), vec.end(), array);
    return array;
  }

  template<template<typename> class Container, typename T>
  T* makeArray(const Container<T>& list)
  {
    std::vector<T> vec (list);
    return makeArray (vec);
  }

  template<template<typename, typename> class Container, typename T, class Allocator>
  T* makeArray(const Container<T, Allocator>& vec, size_t& count)
  {
    T* array = new T [vec.size()];
    std::copy (vec.begin(), vec.end(), array);
    count = vec.size();
    return array;
  }

  template<template<typename> class Container, typename T>
  T* makeArray(const Container<T>& list, size_t& count)
  {
    std::vector<T> vec (list);
    return makeArray (vec, count);
  }

  template<template<typename, typename>class Container, typename T>
  unsigned int getElementsCount(const Container<T, std::allocator<T> >& container) {
    return std::distance(container.begin(), container.end());
  }

  template<template<typename>class Container, typename T>
  unsigned int getElementsCount(const Container<T>& container) {
    return std::distance(container.begin(), container.end());
  }

  template<class SmartPtr, template<typename, typename> class Container, typename T>
  SmartPtr makeSmartPtr(const Container<T, std::allocator<T> >& container) {
    T* array = smartptr_utils::makeArray(container);
    return  SmartPtr(array, smartptr_utils::getElementsCount(container));
  }
}

#endif
