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

#include "Utils.h"

#include "oapHostMatrixUtils.h"

namespace utils {

Compare::Compare() {}

Compare::~Compare() {}

bool Compare::compare(math::ComplexMatrix* matrix, floatt d) {
  if (NULL == matrix) {
    return false;
  }
  uintt length = gRows (matrix) * gColumns (matrix);
  for (uintt fa = 0; fa < length; ++fa) {
    if (!rule(gReValues (matrix)[fa], d)) {
      return false;
    }
  }
  return true;
}

math::ComplexMatrix* create(const math::ComplexMatrix& arg)
{
  return oap::host::NewMatrix(gReValues (&arg) != NULL, gImValues (&arg) != NULL, gColumns (&arg), gRows (&arg));
}

bool AlmostEquals(floatt a, floatt b) {
  return fabs(a - b) < std::numeric_limits<floatt>::epsilon();
}

bool AlmostEquals(floatt a, floatt b, floatt epsilon) {
  return fabs(a - b) < epsilon;
}

void diff(math::ComplexMatrix* output, math::ComplexMatrix* m1, math::ComplexMatrix* m2) {
  for (uintt fa = 0; fa < gColumns (output); ++fa) {
    for (uintt fb = 0; fb < gRows (output); ++fb) {
      uintt index = fa + fb * gColumns (m1);
      if (gReValues (output) != NULL) {
        gReValues (output)[index] = gReValues (m1)[index] - gReValues (m2)[index];
      }
      if (gImValues (output) != NULL) {
        gImValues (output)[index] = gImValues (m1)[index] - gImValues (m2)[index];
      }
    }
  }
}

bool IsEqual(const math::ComplexMatrix& m1, const math::ComplexMatrix& m2, floatt tolerance, math::ComplexMatrix** diff) {
  if (gColumns (&m1) != gColumns (&m2) || gRows (&m1) != gRows (&m2)) {
    return false;
  }
  return HasValues(m1, m2, tolerance, diff);
}

bool HasValues(const math::ComplexMatrix& m1, const math::ComplexMatrix& m2, floatt tolerance, math::ComplexMatrix** diff) {
  if (diff != NULL) {
    (*diff) = NULL;
  }
  if (gColumns (&m1) * gRows (&m1) != gColumns (&m2) * gRows (&m2)) {
    return false;
  }
  bool status = true;
  for (uintt fa = 0; fa < gColumns (&m1); ++fa) {
    for (uintt fb = 0; fb < gRows (&m1); ++fb) {
      uintt index = fa + fb * gColumns (&m1);
      if (gReValues (&m1) != NULL && gReValues (&m2) != NULL) {
        floatt re1 = (gReValues (&m1)[index]);
        floatt re2 = (gReValues (&m2)[index]);

        if (!AlmostEquals(re1, re2, tolerance)) {
          status = false;
          if (diff != NULL) {
            if (*diff == NULL) {
              (*diff) = create(m1);
            }
            SetReIdx (*diff, index, GetReIdx (&m1, index) - GetReIdx (&m2, index));
          }
        }
      }
      if (gImValues (&m1) != NULL && gImValues (&m2) != NULL) {
        floatt im1 = (gImValues (&m1)[index]);
        floatt im2 = (gImValues (&m2)[index]);

        if (!AlmostEquals(im1, im2, tolerance)) {
          status = false;
          if (diff != NULL) {
            if (*diff == NULL) {
              (*diff) = create(m1);
            }
            SetImIdx (*diff, index, GetImIdx (&m1, index) - GetImIdx (&m2, index));
          }
        }
      }
    }
  }
  return status;
}

bool IsIdentityMatrix(const math::ComplexMatrix& m1, floatt tolerance, math::ComplexMatrix** diff)
{
  math::ComplexMatrix* matrix = oap::host::NewMatrixRef (&m1);
  oap::host::SetIdentity(matrix);
  bool isequal = IsEqual(m1, *matrix, tolerance, diff);
  oap::host::DeleteMatrix(matrix);
  return isequal;
}

bool IsDiagonalMatrix(const math::ComplexMatrix& m1, floatt value, floatt tolerance, math::ComplexMatrix** diff)
{
  math::ComplexMatrix* matrix = oap::host::NewMatrixRef (&m1);
  oap::host::SetDiagonalMatrix(matrix, value);
  bool isequal = IsEqual(m1, *matrix, tolerance, diff);
  oap::host::DeleteMatrix(matrix);
  return isequal;
}

bool IsIdentityMatrix (const math::ComplexMatrix& m1, floatt tolerance)
{
  return IsIdentityMatrix (m1, tolerance, NULL);
}

bool IsDiagonalMatrix (const math::ComplexMatrix& m1, floatt value, floatt tolerance)
{
  return IsDiagonalMatrix (m1, value, tolerance, NULL);
}

bool isEqual(const MatrixEx& matrixEx, const uintt* buffer) {
  if (matrixEx.column == buffer[0] && matrixEx.columns == buffer[1] &&
      matrixEx.row == buffer[2] && matrixEx.rows == buffer[3]) {
    return true;
  }
  return false;
}

bool areEqual(math::ComplexMatrix* matrix, floatt value) {
  class CompareImpl : public Compare {
   public:
    bool rule(const floatt& arg1, const floatt& arg2) { return arg1 == arg2; }
  };
  CompareImpl compareImpl;
  return compareImpl.compare(matrix, value);
}

bool areNotEqual(math::ComplexMatrix* matrix, floatt value) {
  class CompareImpl : public Compare {
   public:
    bool rule(const floatt& arg1, const floatt& arg2) { return arg1 != arg2; }
  };
  CompareImpl compareImpl;
  return compareImpl.compare(matrix, value);
}
}
