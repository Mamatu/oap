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

class InfoType : public matrixUtils::Range {
 public:
  static int ELEMENTS;
  static int MEAN;
  static int LARGEST_DIFF;
  static int SMALLEST_DIFF;

 private:
  int m_type;

 public:
  inline InfoType(uintt bcolumn, uintt columns, uintt brow, uintt rows,
                  int type)
      : matrixUtils::Range(bcolumn, columns, brow, rows), m_type(type) {}

  inline InfoType()
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(ELEMENTS) {}

  inline InfoType(int type)
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(type) {}

  inline int getInfo() const { return m_type; }
};

#endif  // INFOTYPE
