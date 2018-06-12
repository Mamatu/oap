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

#ifndef OAP_MATRIXEX_H
#define OAP_MATRIXEX_H

#include "Math.h"

struct MatrixEx {
  uintt beginColumn;
  uintt columnsLength;
  uintt beginRow;
  uintt rowsLength;

  /**
   * @brief boffset - extra offset to dotProduct operation
   */
  uintt boffset;

  /**
   * @brief boffset - extra offset to dotProduct operation
   */
  uintt eoffset;
};

#define erow(matrixex) matrixex.beginRow + matrixex.rowsLength

#define ecolumn(matrixex) matrixex.beginColumn + matrixex.columnsLength

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
  return ::adjust(matrixEx.beginColumn, matrixEx.columnsLength, val);
}

inline bool adjustRows(MatrixEx& matrixEx, uintt val) {
  return ::adjust(matrixEx.beginRow, matrixEx.rowsLength, val);
}

#endif /* MATRIXEX_H */
