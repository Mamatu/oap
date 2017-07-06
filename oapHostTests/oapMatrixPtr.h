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

#include <memory>
#include "Math.h"


namespace {
  using DeleterType = void(*)(math::Matrix*);
}

class MatrixPtr : public std::unique_ptr<math::Matrix, ::DeleterType> {
private:

public:
  MatrixPtr(math::Matrix* matrix) : std::unique_ptr<math::Matrix, ::DeleterType>(matrix, host::DeleteMatrix) {}

  operator math::Matrix*() { return this->get(); }
};

