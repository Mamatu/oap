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

#ifndef OAP_MATRIXSPTR_H
#define OAP_MATRIXSPTR_H

#include <memory>
#include "Math.h"
#include "Matrix.h"

#include "oapSmartPointerUtils.h"

namespace oap
{
namespace stdlib
{
  using MatrixSharedPtr = ::std::shared_ptr<math::Matrix>;

  using MatricesSharedPtr = ::std::shared_ptr<math::Matrix*>;

  using MatrixUniquePtr = ::std::unique_ptr<math::Matrix, deleters::MatrixDeleter>;

  using MatricesUniquePtr = ::std::unique_ptr<math::Matrix*, deleters::MatricesDeleter>;
}

  template<class StdMatrixPtr>
  class MatrixSPtr
  {
    public:
      MatrixSPtr() : StdMatrixPtr()
      {}

      MatrixSPtr(math::Matrix* matrix, deleters::MatrixDeleter deleter) : m_stdMatrixPtr (matrix, deleter) {}

      MatrixSPtr(const MatrixSPtr& orig) = default;
      MatrixSPtr(MatrixSPtr&& orig) = default;
      MatrixSPtr& operator=(const MatrixSPtr& orig) = default;
      MatrixSPtr& operator=(MatrixSPtr&& orig) = default;

      operator math::Matrix*() { return this->get(); }

      math::Matrix* operator->() { return this->get(); }

      void reset ()
      {
        m_stdMatrixPtr.reset ();
      }

      void reset (typename StdMatrixPtr::element_type* t)
      {
        auto& deleter = deleters::get_deleter<deleters::MatrixDeleter, stdlib::MatrixSharedPtr, stdlib::MatrixUniquePtr> (m_stdMatrixPtr);
        reset::reset<stdlib::MatrixSharedPtr, stdlib::MatrixUniquePtr> (m_stdMatrixPtr, deleter, t);
      }

      typename StdMatrixPtr::element_type* get() const
      {
        return m_stdMatrixPtr.get();
      }

    private:
      StdMatrixPtr m_stdMatrixPtr;
  };

  template<class StdMatrixPtr>
  class MatricesSPtr
  {
    public:
      MatricesSPtr() : StdMatrixPtr() {}

      MatricesSPtr(math::Matrix** matrices, size_t count, deleters::MatrixDeleter deleter) :
        m_stdMatrixPtr(matrices, deleters::MatricesDeleter (count, deleter)) {}

      MatricesSPtr(std::initializer_list<math::Matrix*> matrices, deleters::MatrixDeleter deleter) :
         m_stdMatrixPtr (smartptr_utils::makeArray(matrices),
                         deleters::MatricesDeleter (smartptr_utils::getElementsCount (matrices), deleter)) {}

      MatricesSPtr(size_t count, deleters::MatrixDeleter deleter) :
        m_stdMatrixPtr (smartptr_utils::makeArray<math::Matrix*>(count), deleters::MatricesDeleter (count, deleter))  {}

      MatricesSPtr(const MatricesSPtr& orig) = default;
      MatricesSPtr(MatricesSPtr&& orig) = default;
      MatricesSPtr& operator=(const MatricesSPtr& orig) = default;
      MatricesSPtr& operator=(MatricesSPtr&& orig) = default;

      operator math::Matrix**() { return this->get(); }

      math::Matrix*& operator[](size_t index)
      {
        return this->get()[index];
      }

      math::Matrix** operator->() { return this->get(); }

      void reset ()
      {
        m_stdMatrixPtr.reset ();
      }

      template<template<typename, typename> class Container, typename Allocator>
      void reset (const Container<math::Matrix*, Allocator>& matrices)
      {
        size_t newCount = 0;
        auto& deleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);

        math::Matrix** array = smartptr_utils::makeArray<Container, math::Matrix*, Allocator> (matrices, newCount);
        reset::reset<stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr, deleter, array);

        auto& ndeleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ndeleter.setCount (newCount);
      }

      template<template<typename> class Container>
      void reset (const Container<math::Matrix*>& matrices)
      {
        size_t newCount = 0;
        auto& deleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);

        math::Matrix** array = smartptr_utils::makeArray<Container, math::Matrix* >(matrices, newCount);
        reset::reset<stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr, deleter, array);

        auto& ndeleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ndeleter.setCount (newCount);
      }

      void reset (typename StdMatrixPtr::element_type* t, size_t count)
      {
        auto& deleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);

        reset::reset<stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr, deleter, t);

        auto& ndeleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ndeleter.setCount (count);
      }

      typename StdMatrixPtr::element_type* get() const
      {
        return m_stdMatrixPtr.get();
      }

    private:
      StdMatrixPtr m_stdMatrixPtr;
  };
}

namespace oap
{

using MatrixSharedPtr = MatrixSPtr<stdlib::MatrixSharedPtr>;

using MatricesSharedPtr = MatricesSPtr<stdlib::MatricesSharedPtr>;

using MatrixUniquePtr = MatrixSPtr<stdlib::MatrixUniquePtr>;

using MatricesUniquePtr = MatricesSPtr<stdlib::MatricesUniquePtr>;

}

#endif

