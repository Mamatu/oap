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

#ifndef OAP_INFOTYPE
#define OAP_INFOTYPE

#include "MatrixUtils.h"
#include <limits>

namespace
{
  const floatt g_tolerance = 0.0001f;
}

class InfoType : public matrixUtils::Range {
 public:
  static int ELEMENTS;
  static int MEAN;
  static int LARGEST_DIFF;
  static int SMALLEST_DIFF;

 private:
  int m_type;
  floatt m_tolerance = g_tolerance;

 public:
  inline InfoType(uintt bcolumn, uintt columns, uintt brow, uintt rows,
                  int type, floatt tolerance = g_tolerance)
      : matrixUtils::Range(bcolumn, columns, brow, rows), m_type(type), m_tolerance (tolerance) {}

  inline InfoType(floatt tolerance = g_tolerance)
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(ELEMENTS), m_tolerance (tolerance) {}

  inline InfoType(int type, floatt tolerance = g_tolerance)
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(type), m_tolerance (tolerance) {}

  inline int getInfo() const { return m_type; }
  inline floatt getTolerance() const { return m_tolerance; }
};

#endif  // INFOTYPE
