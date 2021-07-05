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

#ifndef OAP_MATRIX_H
#define OAP_MATRIX_H

#include "Math.hpp"
#include "oapMemory.hpp"

//#define DEBUG
namespace math
{
struct MatrixDim
{
  uintt columns;
  uintt rows;
};

struct MatrixLoc
{
  uintt x;
  uintt y;
};

struct Matrix
{
  oap::Memory mem;
  oap::MemoryRegion reg;
};

/**
 * Columns orientation
 */
struct ComplexMatrix
{
  Matrix re;
  Matrix im;
  MatrixDim dim;
};
}

namespace oap
{
namespace math
{
  using MatrixDim = ::math::MatrixDim;
  using MatrixLoc = ::math::MatrixLoc;
  using Matrix = ::math::Matrix;
  using ComplexMatrix = ::math::ComplexMatrix;
}
}

#define OAP_REGION_IS_VALID(region) (!(region.dims.width == 0 || region.dims.height == 0))

#endif
