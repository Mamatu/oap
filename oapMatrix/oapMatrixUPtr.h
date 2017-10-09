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

#ifndef MATRIXUPTR_H
#define MATRIXUPTR_H

#include <memory>
#include "Math.h"
#include "Matrix.h"

#include "oapSmartPointerUtils.h"

namespace oap {
  class MatrixUPtr : public std::unique_ptr<math::Matrix, deleters::MatrixDeleter> {
    public:
      MatrixUPtr(math::Matrix* matrix, deleters::MatrixDeleter deleter) : std::unique_ptr<math::Matrix, deleters::MatrixDeleter>(matrix, deleter) {}

      operator math::Matrix*() { return this->get(); }

      math::Matrix* operator->() { return this->get(); }
  };

  class MatricesUPtr : public std::unique_ptr<math::Matrix*, deleters::MatricesDeleter> {
    public:
      MatricesUPtr(math::Matrix** matrices, deleters::MatricesDeleter deleter) :
        std::unique_ptr<math::Matrix*, deleters::MatricesDeleter>(matrices, deleter) {}

      MatricesUPtr(std::initializer_list<math::Matrix*> matrices, deleters::MatricesDeleter deleter) :
        std::unique_ptr<math::Matrix*, deleters::MatricesDeleter>(smartptr_utils::makeArray(matrices), deleter) {}

      operator math::Matrix**() { return this->get(); }

      math::Matrix** operator->() { return this->get(); }
  };
}

#endif

