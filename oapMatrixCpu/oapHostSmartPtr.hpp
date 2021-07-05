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

#ifndef OAP_HOST_SMART_PTR_HPP
#define OAP_HOST_SMART_PTR_HPP

#include <type_traits>

#include "oapHostComplexMatrixApi.hpp"
#include "oapHostMatrixApi.hpp"
#include "Math.hpp"
#include "Matrix.hpp"

#include "oapMatrixSPtr.hpp"

namespace oap {
namespace host {

namespace
{
struct ComplexMatrixDeleter
{
  ComplexMatrixDeleter(const math::ComplexMatrix* matrix)
  {
    oap::chost::DeleteComplexMatrix (matrix);
  }
};

struct MatrixDeleter
{
  MatrixDeleter(const math::Matrix* matrix)
  {
    oap::host::DeleteMatrix (matrix);
  }
};
}

template<typename MatrixT>
void DeleteMatrixWrapper (const MatrixT* matrix, void* thiz)
{
  logTrace("Destroy: HostComplexMatrixPtr = %p matrix = %p", thiz, matrix);
  constexpr bool isComplexMatrix = std::is_same<MatrixT, math::ComplexMatrix>::value;
  typename std::conditional<isComplexMatrix, ComplexMatrixDeleter, MatrixDeleter>::type obj(matrix);
}

template<typename MatrixT>
void DeleteMatricesWrapper (MatrixT** matrices, size_t count, void* thiz)
{
  for (size_t idx = 0; idx < count; ++idx)
  {
    DeleteMatrixWrapper<MatrixT> (matrices[idx], thiz);
  }
}

}
}

#endif
