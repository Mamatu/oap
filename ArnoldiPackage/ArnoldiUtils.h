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

#ifndef ARNOLDIUTILS_H
#define ARNOLDIUTILS_H

#include "Math.h"
#include "Matrix.h"

#include <cstddef>

class EigenPair {
  private:
    Complex m_evalue;
    const math::ComplexMatrix* m_matrix;
    uint m_index;
  public:

    EigenPair (floatt revalue, const math::ComplexMatrix* matrix = nullptr, uint index = 0) :
      m_evalue(revalue), m_matrix(matrix), m_index(index)
    {
    }

    EigenPair(const Complex& evalue, const math::ComplexMatrix* matrix, uint index = 0) :
      m_evalue(evalue), m_matrix(matrix), m_index(index)
    {}

    EigenPair(const Complex& evalue, uint index) :
      m_evalue(evalue), m_matrix(NULL), m_index(index)
    {}

    Complex getEValue() const { return m_evalue; }

    floatt re() const { return m_evalue.re; }
    floatt im() const { return m_evalue.im; }

    const math::ComplexMatrix* getMatrix() const { return m_matrix; }

    uint getIndex() const { return  m_index; }
};

namespace ArnUtils {

bool SortLargestValues(const EigenPair& i, const EigenPair& j);

bool SortLargestReValues(const EigenPair& i, const EigenPair& j);

bool SortLargestImValues(const EigenPair& i, const EigenPair& j);

bool SortSmallestValues(const EigenPair& i, const EigenPair& j);

bool SortSmallestReValues(const EigenPair& i, const EigenPair& j);

bool SortSmallestImValues(const EigenPair& i, const EigenPair& j);

typedef bool (*SortType)(const EigenPair& i, const EigenPair& j);

enum CheckType {
  CHECK_INTERNAL,
  CHECK_FIRST_STOP,
  CHECK_COUNTER
};

enum TriangularHProcedureType {
  CALC_IN_DEVICE,
  CALC_IN_HOST,
  CALC_IN_DEVICE_STEP
};

enum Type { UNDEFINED, DEVICE, HOST };
}
#endif  // ARNOLDIUTILS_H
