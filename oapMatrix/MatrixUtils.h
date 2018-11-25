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

#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H

#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <utility>
#include "Matrix.h"
#include "MatrixAPI.h"

namespace matrixUtils
{

extern const char* ID_COLUMNS;
extern const char* ID_ROWS;
extern const char* ID_LENGTH;

template <typename T>
class OccurencesList : public std::vector<std::pair<uintt, T> > {};

template <typename T>
class SubArrays : public std::vector<std::pair<T*, uintt> > {};

class Range {
 protected:
  uintt m_bcolumn;
  uintt m_columns;
  uintt m_brow;
  uintt m_rows;

 public:
  Range(uintt bcolumn, uintt columns, uintt brow, uintt rows);

  virtual ~Range();

  virtual uintt getBColumn() const;
  virtual uintt getEColumn() const;
  virtual uintt getColumns() const;

  virtual uintt getBRow() const;
  virtual uintt getERow() const;
  virtual uintt getRows() const;
};

class MatrixRange : public Range {
 protected:
  const math::Matrix* m_matrix;

  void getSubArrays(SubArrays<floatt>& subArrays, floatt* array,
                    const math::Matrix* matrix) const;

 public:
  MatrixRange(const math::Matrix* matrix);
  MatrixRange(const math::Matrix* matrix, const Range& range);
  MatrixRange(const math::Matrix* matrix, uintt bcolumn, uintt columns,
              uintt brow, uintt rows);

  virtual ~MatrixRange();

  bool isReValues() const;
  bool isImValues() const;

  virtual uintt getEColumn() const;
  virtual uintt getERow() const;

  const math::Matrix* getMatrix() const;

  void getReSubArrays(SubArrays<floatt>& subArrays) const;
  void getImSubArrays(SubArrays<floatt>& subArrays) const;
};

template<typename T>
class PrintArgs
{
  public:
    T zeroLimit;
    bool repeats;
    std::string sectionSeparator;
    size_t sectionLength;

    PrintArgs(T _zeroLimit = 0, bool _repeats = false, const std::string& _sectionSeparator = "|\n", size_t _sectionLength = std::numeric_limits<size_t>::max()):
      zeroLimit(_zeroLimit), repeats(_repeats), sectionSeparator(_sectionSeparator), sectionLength(_sectionLength)
    {}
};

template <typename T>
void PrepareOccurencesList (OccurencesList<T>& occurencesList, T* array, uintt length, const PrintArgs<T>& args = PrintArgs<T>())
{
  const T zeroLimit = args.zeroLimit;
  const bool repeats = args.repeats;
  const size_t sectionLength = args.sectionLength;

  for (uintt fa = 0; fa < length; ++fa)
  {
    floatt value = array[fa];
    if (-zeroLimit < value && value < zeroLimit)
    {
      value = 0.f;
    }
    if (repeats == true || occurencesList.size() == 0 || value != occurencesList[occurencesList.size() - 1].second)
    {
      uintt a = 1;
      occurencesList.push_back(std::make_pair<uintt&, floatt&>(a, value));
    }
    else
    {
      occurencesList[occurencesList.size() - 1].first++;
    }
  }
}

};

#endif  // MATRIXUTILS_H
