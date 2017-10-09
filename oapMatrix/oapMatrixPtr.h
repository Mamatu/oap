/*
 * Copyright 2016, 2017 Marcin Matula
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

#ifndef MATRIXPTR_H
#define MATRIXPTR_H

#include <memory>
#include "Math.h"
#include "Matrix.h"

#include "oapSmartPointerUtils.h"

namespace oap {
  class MatrixPtr : public std::shared_ptr<math::Matrix> {
    public:
      MatrixPtr(math::Matrix* matrix, deleters::MatrixDeleter deleter) : std::shared_ptr<math::Matrix>(matrix, deleter) {}

      operator math::Matrix*() { return this->get(); }

      math::Matrix* operator->() { return this->get(); }
  };

  class MatricesPtr : public std::shared_ptr<math::Matrix*> {
    public:
      MatricesPtr(math::Matrix** matrices, deleters::MatricesDeleter deleter) : std::shared_ptr<math::Matrix*>(matrices, deleter) {}

      MatricesPtr(std::initializer_list<math::Matrix*> matrices, deleters::MatricesDeleter deleter) :
        std::shared_ptr<math::Matrix*>(smartptr_utils::makeArray(matrices), deleter) {}

      operator math::Matrix**() { return this->get(); }

      math::Matrix** operator->() { return this->get(); }
  };
}

#endif

