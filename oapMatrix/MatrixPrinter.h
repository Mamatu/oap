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

#ifndef OAP_MATRIX_PRINTER_H
#define OAP_MATRIX_PRINTER_H

#include "MatrixUtils.h"

namespace matrixUtils
{

template <typename T>
void PrintArrays(std::string& output, T** arrays, uintt* lengths, uintt count, const PrintArgs& args = PrintArgs())
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;
  const std::string sectionSeparator = args.section.separator;
  const size_t sectionLength = args.section.length;

  output = "";

  OccurencesList<T> valuesVec;
  uintt totalLength = 0;
  for (uintt index = 0; index < count; ++index)
  {
    T* array = arrays[index];
    uintt length = lengths[index];
    PrepareOccurencesList (valuesVec, array, length, args);
    totalLength += length;
  }

  std::stringstream sstream;
  sstream << args.pretext;
  sstream << "[";

  for (uintt fa = 0, fa1 = 0; fa < valuesVec.size(); ++fa)
  {
    sstream << valuesVec[fa].second;
    const uintt count = valuesVec[fa].first;
    if (count > 1)
    {
      sstream << " <repeats " << count << " times>";
    }
    fa1 += count;
    bool lastPosition = (fa1 == totalLength);
    if (!lastPosition)
    {
      bool endLine = (fa1 % sectionLength) == 0;
      if (!endLine)
      {
        sstream << ", ";
      }
      if (endLine)
      {
        sstream << sectionSeparator;
      }
    }
  }

  sstream << "]";
  sstream << args.posttext;
  output += sstream.str();
}

template <typename T>
void PrintArrays(std::string& output, const SubArrays<T>& subArrays, const PrintArgs& args = PrintArgs())
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;

  uintt count = subArrays.size();
  T** arrays = new T* [count];
  uintt* lengths = new uintt[count];

  for (uintt fa = 0; fa < count; ++fa)
  {
    arrays[fa] = subArrays[fa].first;
    lengths[fa] = subArrays[fa].second;
  }

  PrintArrays(output, arrays, lengths, count, args);
  delete[] arrays;
  delete[] lengths;
}

template <typename T>
void PrintArray(std::string& output, T* array, uintt length, const PrintArgs& args = PrintArgs())
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;

  T* arrays[] = {array};
  uintt lengths[] = {length};
  PrintArrays(output, arrays, lengths, 1, args);
}

void PrintReValues (std::string& output, const MatrixRange& matrixRange, const PrintArgs& args = PrintArgs());

void PrintImValues (std::string& output, const MatrixRange& matrixRange, const PrintArgs& args = PrintArgs());

void PrintMatrix (std::string& output, const MatrixRange& matrixRange, const PrintArgs& args = PrintArgs());

void PrintMatrix (std::string& output, const math::Matrix* matrix, const PrintArgs& args = PrintArgs());

}

#endif
