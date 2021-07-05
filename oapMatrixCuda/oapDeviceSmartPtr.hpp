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

#ifndef OAP_DEVICE_SMART_PTR_H
#define OAP_DEVICE_SMART_PTR_H

#include "oapCudaMatrixUtils.hpp"
#include "Math.hpp"
#include "Matrix.hpp"

#include "oapMatrixSPtr.hpp"

namespace oap {
namespace device {

template<typename MatrixT, typename Thiz>
void DeleteMatrixWrapper (const MatrixT* matrix, Thiz* thiz)
{
  logTrace("Destroy: HostComplexMatrixPtr = %p matrix = %p", thiz, matrix);
  oap::cuda::DeleteDeviceMatrix (matrix);
}

template<typename MatrixT, typename Thiz>
void DeleteMatricesWrapper (MatrixT** matrices, size_t count, Thiz* thiz)
{
  for (size_t idx = 0; idx < count; ++idx)
  {
    DeleteMatrixWrapper<MatrixT> (matrices[idx], thiz);
  }
}

}
}

#endif
