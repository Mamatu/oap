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

#include "Utils.h"

#include "oapHostMatrixUtils.h"

namespace utils {

Compare::Compare() {}

Compare::~Compare() {}

bool Compare::compare(math::Matrix* matrix, floatt d) {
  if (NULL == matrix) {
    return false;
  }
  uintt length = matrix->rows * matrix->columns;
  for (uintt fa = 0; fa < length; ++fa) {
    if (!rule(matrix->reValues[fa], d)) {
      return false;
    }
  }
  return true;
}

math::Matrix* create(const math::Matrix& arg) {
  return oap::host::NewMatrix(arg.reValues != NULL, arg.imValues != NULL,
                         arg.columns, arg.rows);
}

bool AlmostEquals(floatt a, floatt b) {
  return fabs(a - b) < std::numeric_limits<floatt>::epsilon();
}

bool AlmostEquals(floatt a, floatt b, floatt epsilon) {
  return fabs(a - b) < epsilon;
}

void diff(math::Matrix* output, math::Matrix* m1, math::Matrix* m2) {
  for (uintt fa = 0; fa < output->columns; ++fa) {
    for (uintt fb = 0; fb < output->rows; ++fb) {
      uintt index = fa + fb * m1->columns;
      if (output->reValues != NULL) {
        output->reValues[index] = m1->reValues[index] - m2->reValues[index];
      }
      if (output->imValues != NULL) {
        output->imValues[index] = m1->imValues[index] - m2->imValues[index];
      }
    }
  }
}

bool IsEqual(const math::Matrix& m1, const math::Matrix& m2,
             math::Matrix** diff) {
  if (m1.columns != m2.columns || m1.rows != m2.rows) {
    return false;
  }
  return HasValues(m1, m2, diff);
}

bool HasValues(const math::Matrix& m1, const math::Matrix& m2,
               math::Matrix** diff) {
  if (diff != NULL) {
    (*diff) = NULL;
  }
  if (m1.columns * m1.rows != m2.columns * m2.rows) {
    return false;
  }
  bool status = true;
  for (uintt fa = 0; fa < m1.columns; ++fa) {
    for (uintt fb = 0; fb < m1.rows; ++fb) {
      uintt index = fa + fb * m1.columns;
      if (m1.reValues != NULL && m2.reValues != NULL) {
        floatt re1 = (m1.reValues[index]);
        floatt re2 = (m2.reValues[index]);

        if (!AlmostEquals(re1, re2, 0.0001)) {
          status = false;
          if (diff != NULL) {
            if (*diff == NULL) {
              (*diff) = create(m1);
            }
            (*diff)->reValues[index] = m1.reValues[index] - m2.reValues[index];
          }
        }
      }
      if (m1.imValues != NULL && m2.imValues != NULL) {
        floatt im1 = (m1.imValues[index]);
        floatt im2 = (m2.imValues[index]);

        if (!AlmostEquals(im1, im2, 0.001)) {
          status = false;
          if (diff != NULL) {
            if (*diff == NULL) {
              (*diff) = create(m1);
            }
            (*diff)->imValues[index] = m1.imValues[index] - m2.imValues[index];
          }
        }
      }
    }
  }
  return status;
}

bool IsIdentityMatrix(const math::Matrix& m1, math::Matrix** output) {
  math::Matrix* matrix = oap::host::NewMatrixRef (&m1);
  oap::host::SetIdentity(matrix);
  bool isequal = IsEqual(m1, *matrix, output);
  oap::host::DeleteMatrix(matrix);
  return isequal;
}

bool IsDiagonalMatrix(const math::Matrix& m1, floatt value,
                      math::Matrix** output) {
  math::Matrix* matrix = oap::host::NewMatrixRef (&m1);
  oap::host::SetDiagonalMatrix(matrix, value);
  bool isequal = IsEqual(m1, *matrix, output);
  oap::host::DeleteMatrix(matrix);
  return isequal;
}

bool IsIdentityMatrix(const math::Matrix& m1) {
  return IsIdentityMatrix(m1, NULL);
}

bool IsDiagonalMatrix(const math::Matrix& m1, floatt value) {
  return IsDiagonalMatrix(m1, value, NULL);
}

bool isEqual(const MatrixEx& matrixEx, const uintt* buffer) {
  if (matrixEx.beginColumn == buffer[0] && matrixEx.columnsLength == buffer[1] &&
      matrixEx.beginRow == buffer[2] && matrixEx.rowsLength == buffer[3]) {
    return true;
  }
  return false;
}

bool areEqual(math::Matrix* matrix, floatt value) {
  class CompareImpl : public Compare {
   public:
    bool rule(const floatt& arg1, const floatt& arg2) { return arg1 == arg2; }
  };
  CompareImpl compareImpl;
  return compareImpl.compare(matrix, value);
}

bool areNotEqual(math::Matrix* matrix, floatt value) {
  class CompareImpl : public Compare {
   public:
    bool rule(const floatt& arg1, const floatt& arg2) { return arg1 != arg2; }
  };
  CompareImpl compareImpl;
  return compareImpl.compare(matrix, value);
}
}
