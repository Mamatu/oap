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

  using MatrixUniquePtr = ::std::unique_ptr<math::Matrix, deleters::MatrixDeleterWrapper>;

  using MatricesUniquePtr = ::std::unique_ptr<math::Matrix*, deleters::MatricesDeleter>;
}

  template<class StdMatrixPtr>
  class MatrixSPtr
  {
    public:
      using Type = typename StdMatrixPtr::element_type;

      MatrixSPtr () = delete;
      MatrixSPtr (Type* matrix, deleters::MatrixDeleter deleter, bool bDeallocate = true);

      MatrixSPtr(const MatrixSPtr& orig) = default;
      MatrixSPtr(MatrixSPtr&& orig) = default;
      MatrixSPtr& operator=(const MatrixSPtr& orig) = default;
      MatrixSPtr& operator=(MatrixSPtr&& orig) = default;

      virtual ~MatrixSPtr()
      {
        // empty
      }

      operator Type*() { return this->get(); }

      Type* operator->() const { return this->get(); }

      void reset ()
      {
        m_stdMatrixPtr.reset ();
      }

      void reset (Type* t, bool bDeallocate = true)
      {
        reset::reset<stdlib::MatrixSharedPtr, stdlib::MatrixUniquePtr> (m_stdMatrixPtr, bDeallocate, t);
      }

      Type* get() const
      {
        return m_stdMatrixPtr.get();
      }

    private:
      StdMatrixPtr m_stdMatrixPtr;
  };

  template<class StdMatrix>
  MatrixSPtr<StdMatrix>::MatrixSPtr (Type* matrix, deleters::MatrixDeleter deleter, bool bDeallocate) :
    m_stdMatrixPtr (matrix, [deleter, bDeallocate]() -> deleters::MatrixDeleterWrapper { return deleters::MatrixDeleterWrapper(bDeallocate, deleter); }())
  {
  }

  template<class StdMatrixPtr>
  class MatricesSPtr
  {
    public:
      using ArrayType = typename StdMatrixPtr::element_type;
      using Type = typename std::remove_pointer<ArrayType>::type;

      MatricesSPtr (ArrayType* matrices, size_t count, deleters::MatrixDeleter deleter, bool bCopyArray = true) :
        m_stdMatrixPtr( bCopyArray ? smartptr_utils::makeArray (matrices, count) : matrices, deleters::MatricesDeleter (count, deleter)) {}

      MatricesSPtr (std::initializer_list<ArrayType> matrices, deleters::MatrixDeleter deleter) :
         m_stdMatrixPtr (smartptr_utils::makeArray (matrices).ptr,
                         deleters::MatricesDeleter (smartptr_utils::getElementsCount (matrices), deleter)) {}

      MatricesSPtr (size_t count, deleters::MatrixDeleter deleter) :
        m_stdMatrixPtr (smartptr_utils::makeArray<ArrayType> (count), deleters::MatricesDeleter (count, deleter))  {}

      MatricesSPtr(const MatricesSPtr& orig) = default;
      MatricesSPtr(MatricesSPtr&& orig) = default;
      MatricesSPtr& operator=(const MatricesSPtr& orig) = default;
      MatricesSPtr& operator=(MatricesSPtr&& orig) = default;

      virtual ~MatricesSPtr()
      {
        // empty
      }

      operator ArrayType*() { return this->get(); }

      ArrayType& operator[](size_t index)
      {
        return this->get()[index];
      }

      ArrayType* operator->() const { return this->get(); }

      void reset ()
      {
        m_stdMatrixPtr.reset ();
      }

      template<template<typename, typename> class Container, typename Allocator>
      void reset (const Container<ArrayType, Allocator>& matrices)
      {
        using ArrayPtr = smartptr_utils::ArrayPtr<ArrayType>;

        auto& deleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);

        ArrayPtr array = smartptr_utils::makeArray<Container, ArrayType, Allocator> (matrices);
        reset::reset<stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr, deleter, array.ptr);

        size_t newCount = array.count;

        auto& ndeleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ndeleter.setCount (newCount);
      }

      template<template<typename> class Container>
      void reset (const Container<ArrayType>& matrices)
      {
        using ArrayPtr = smartptr_utils::ArrayPtr<ArrayType>;

        auto& deleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);

        ArrayPtr array = smartptr_utils::makeArray<Container, ArrayType>(matrices);
        reset::reset<stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr, deleter, array.ptr);

        size_t newCount = array.count;

        auto& ndeleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ndeleter.setCount (newCount);
      }

      void reset (ArrayType* t, size_t count)
      {
        auto& deleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ArrayType* array = smartptr_utils::makeArray<ArrayType> (t, count);

        reset::reset<stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr, deleter, array);

        auto& ndeleter = deleters::get_deleter<deleters::MatricesDeleter, stdlib::MatricesSharedPtr, stdlib::MatricesUniquePtr> (m_stdMatrixPtr);
        ndeleter.setCount (count);
      }

      ArrayType* get() const
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

