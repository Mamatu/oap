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

#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H

#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <utility>
#include <functional>
#include "Matrix.hpp"
#include "MatrixAPI.hpp"
#include "oapRange.hpp"

namespace matrixUtils
{

extern const char* ID_COLUMNS;
extern const char* ID_ROWS;
extern const char* ID_LENGTH;

template <typename T>
class OccurencesList : public std::vector<std::pair<uintt, T> > {};

template <typename T>
class SubArrays : public std::vector<std::pair<T*, uintt> > {};

class MatrixRange : public Range {
 protected:
  const math::ComplexMatrix* m_matrix;

  void getSubArrays(SubArrays<floatt>& subArrays, floatt* array,
                    const math::ComplexMatrix* matrix) const;

 public:
  MatrixRange(const math::ComplexMatrix* matrix);
  MatrixRange(const math::ComplexMatrix* matrix, const Range& range);
  MatrixRange(const math::ComplexMatrix* matrix, uintt bcolumn, uintt columns,
              uintt brow, uintt rows);

  virtual ~MatrixRange();

  bool isReValues() const;
  bool isImValues() const;

  virtual uintt getEColumn() const;
  virtual uintt getERow() const;

  const math::ComplexMatrix* getMatrix() const;

  void getReSubArrays(SubArrays<floatt>& subArrays) const;
  void getImSubArrays(SubArrays<floatt>& subArrays) const;
};

class Section
{
  public:
    std::string separator = "|\n";
    size_t length = std::numeric_limits<size_t>::max(); ///< length of section, after that will be printed separator

    Section ();

    Section (const std::string& _separator, size_t _length);
};

class PrintArgs
{
  public:
    std::string pretext = ""; ///<text printed before matrix string representation
    std::string posttext = ""; ///<text printed after matrix string representation

    std::string leftBracket = "[";
    std::string rightBracket = "]";
    floatt zrr = 0; ///< zero round range any number which fullfils condition |number| <= zrr, will be print as zero
    bool repeats = false; ///< if true the same number will be repeatedly printed, otherwise will be used pattern <repeats x times>

    Section section;

    std::string postRe = " + ";
    std::string preIm = "i * ";

    bool printIndex = false;

    size_t floatPrecision = 9;

    enum class FloatPrintMode
    {
      SCIENTIFIC_NOTATION,
      FIXED,
      NORMAL
    };

    FloatPrintMode floatPrintMode = FloatPrintMode::FIXED;

    PrintArgs ();

    PrintArgs (const std::string& _pretext, const std::string& _posttext, floatt _zrr, bool _repeats, const std::string& _sectionSeparator, size_t _sectionLength);

    PrintArgs (floatt _zrr, bool _repeats);

    PrintArgs (floatt _zrr, bool _repeats, const std::string& _sectionSeparator, size_t _sectionLength);

    PrintArgs (const std::string& _pretext, const std::string& _posttext);

    PrintArgs (const std::string& _pretext);

    PrintArgs (const char* _pretext);

    void setReImSeparator (const std::string& _postRe, const std::string& _preIm);

    inline void prepareSection (const math::ComplexMatrix* matrix)
    {
      section.length = gColumns (matrix);
    }
};

using ValueCallback = std::function<void(floatt)>;

template <typename T>
void PrepareOccurencesList (OccurencesList<T>& occurencesList, T* array, uintt length, const PrintArgs& args = PrintArgs(), ValueCallback&& valueCallback = [](floatt){})
{
  const T zrr = args.zrr;
  const bool repeats = args.repeats;

  for (uintt fa = 0; fa < length; ++fa)
  {
    floatt value = array[fa];
    if (-zrr < value && value < zrr)
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

    valueCallback (value);
  }
}

};

#endif  // MATRIXUTILS_H
