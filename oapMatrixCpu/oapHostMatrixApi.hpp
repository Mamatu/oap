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

#ifndef OAP_HOST_MATRIX_API_H
#define OAP_HOST_MATRIX_API_H

#include "Matrix.hpp"
#include "oapGenericMatrixApi.hpp"

namespace oap
{
namespace host
{

void SetValueToMatrix (math::Matrix* matrix, floatt v);
void SetZeroMatrix (math::Matrix* matrix);

math::Matrix GetRefHostMatrix (const math::Matrix* matrix);
math::MatrixInfo GetMatrixInfo (const math::Matrix* matrix);
oap::MemoryLoc GetMatrixMemoryLoc (const math::Matrix* matrix);
oap::MemoryRegion GetMatrixMemoryRegion (const math::Matrix* matrix);
math::Matrix* NewMatrixWithValue (uintt columns, uintt rows, floatt value);
math::Matrix* NewMatrix (uintt columns, uintt rows);

void DeleteMatrix (const math::Matrix* matrix);

uintt GetColumns (const math::Matrix* matrix);
uintt GetRows (const math::Matrix* matrix);
floatt GetValue (const math::Matrix* matrix, uintt column, uintt row);

void CopyHostMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src);
}
}

#endif /* OAP_HOST_MATRIX_UTILS_H */
