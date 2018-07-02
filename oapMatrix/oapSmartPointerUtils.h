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
#include <initializer_list>
#include <iterator>
#include <memory>
#include <vector>

#include "Matrix.h"

namespace deleters
{

  using MatrixDeleter = void(*)(const math::Matrix*);

  class MatricesDeleter {
      unsigned int m_count;
      MatrixDeleter m_deleter;
    public:
      MatricesDeleter(unsigned int count, deleters::MatrixDeleter deleter) : 
        m_count(count), m_deleter(deleter) {}

      MatricesDeleter& operator() (math::Matrix** matrices) {
        for (unsigned int idx = 0; idx < m_count; ++idx) {
          m_deleter(matrices[idx]);
        }
        delete[] matrices;
        return *this;
      }
  };
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

  template<template<typename, typename> class Container, typename T>
  T* makeArray(const Container<T, std::allocator<T> >& vec) {
    T* array = new T[vec.size()];
    std::copy(vec.begin(), vec.end(), array);
    return array;
  }

  template<template<typename> class Container, typename T>
  T* makeArray(const Container<T>& list) {
    std::vector<T> vec(list);
    return makeArray(vec);
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
