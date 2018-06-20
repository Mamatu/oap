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

  template<class StdMatrixPtr>
  class MatrixSPtr : public StdMatrixPtr {
    public:
      MatrixSPtr() : StdMatrixPtr()
      {}

      MatrixSPtr(math::Matrix* matrix, deleters::MatrixDeleter deleter) : StdMatrixPtr(matrix, deleter) {}

      operator math::Matrix*() { return this->get(); }

      math::Matrix* operator->() { return this->get(); }
  };

  template<class StdMatrixPtr>
  class MatricesSPtr : public StdMatrixPtr {
    public:
      MatricesSPtr() : StdMatrixPtr() {}

      MatricesSPtr(math::Matrix** matrices, deleters::MatricesDeleter deleter) : StdMatrixPtr(matrices, deleter) {}

      MatricesSPtr(std::initializer_list<math::Matrix*> matrices, deleters::MatricesDeleter deleter) :
        StdMatrixPtr(smartptr_utils::makeArray(matrices), deleter) {}

      MatricesSPtr(size_t count, deleters::MatricesDeleter deleter) :
        StdMatrixPtr(smartptr_utils::makeArray<math::Matrix*>(count), deleter) {}

      operator math::Matrix**() { return this->get(); }

      math::Matrix*& operator[](size_t index)
      {
        return this->get()[index];
      }

      math::Matrix** operator->() { return this->get(); }
  };
}

namespace oap
{
namespace stdlib
{
  using MatrixSharedPtr = ::std::shared_ptr<math::Matrix>;

  using MatricesSharedPtr = ::std::shared_ptr<math::Matrix*>;

  using MatrixUniquePtr = ::std::unique_ptr<math::Matrix, deleters::MatrixDeleter>;

  using MatricesUniquePtr = ::std::unique_ptr<math::Matrix*, deleters::MatricesDeleter>;
}

using MatrixSharedPtr = MatrixSPtr<stdlib::MatrixSharedPtr>;

using MatricesSharedPtr = MatricesSPtr<stdlib::MatricesSharedPtr>;

using MatrixUniquePtr = MatrixSPtr<stdlib::MatrixUniquePtr>;

using MatricesUniquePtr = MatricesSPtr<stdlib::MatricesUniquePtr>;

}

#endif

