/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef OAP_MATRIX_EX_H
#define OAP_MATRIX_EX_H

#include "Math.h"
#include "Matrix.h"

struct MatrixEx
{
  uintt column;
  uintt columns;

  uintt row;
  uintt rows;
};

#define mex_erow(matrixex) matrixex.row + matrixex.rows

#define mex_ecolumn(matrixex) matrixex.column + matrixex.columns

namespace {
  bool adjust(uintt& v1, uintt& v2, uintt val) {
    if (v1 > val) { return false; }
    if (v1 + v2 > val) {
      v2 = val - v1;
    }
    return true;
  }
}

inline bool adjustColumns(MatrixEx& matrixEx, uintt val) {
  return ::adjust(matrixEx.column, matrixEx.columns, val);
}

inline bool adjustRows(MatrixEx& matrixEx, uintt val) {
  return ::adjust(matrixEx.row, matrixEx.rows, val);
}

#endif /* MATRIXEX_H */
