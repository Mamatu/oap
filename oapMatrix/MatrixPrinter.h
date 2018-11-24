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

#ifndef OAP_MATRIX_PRINTER_H
#define OAP_MATRIX_PRINTER_H

#include "MatrixUtils.h"

namespace matrixUtils
{

template<typename T>
class PrintArgs
{
  public:
    T zeroLimit;
    bool repeats;
    bool pipe;
    bool endl;
    size_t sectionLength;

    PrintArgs(T _zeroLimit = 0, bool _repeats = true, bool _pipe = true, bool _endl = true, size_t _sectionLength = std::numeric_limits<size_t>::max()):
      zeroLimit(_zeroLimit), repeats(_repeats), pipe(_pipe), endl(_endl), sectionLength(_sectionLength)
    {}
};

template <typename T>
void PrintArrays(std::string& output, T** arrays, uintt* lengths, uintt count, const PrintArgs<T>& args = PrintArgs<T>())
{
  T zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;
  size_t sectionLength = args.sectionLength;

  output = "[";
  OccurencesList<T> valuesVec;
  uintt totalLength = 0;
  for (uintt index = 0; index < count; ++index)
  {
    T* array = arrays[index];
    uintt length = lengths[index];
    PrepareOccurencesList(valuesVec, array, length, repeats, zeroLimit,
                          sectionLength, totalLength);
    totalLength += length;
  }
  std::stringstream sstream;
  for (uintt fa = 0, fa1 = 0; fa < valuesVec.size(); ++fa)
  {
    sstream << valuesVec[fa].second;
    const uintt count = valuesVec[fa].first;
    if (count > 1)
    {
      sstream << " <repeats " << count << " times>";
    }
    fa1 += count;
    bool lastPosition = fa1 == totalLength;
    if (!lastPosition)
    {
      bool endLine = (fa1 % sectionLength) == 0;
      if (!endLine)
      {
        sstream << ", ";
      } else if (!pipe && endLine)
      {
        sstream << ", ";
      } else if (pipe && endLine)
      {
        sstream << " | ";
      }
      if (endLine && endl)
      {
        sstream << std::endl;
      }
    }
    output += sstream.str();
    sstream.str(std::string());
  }
  sstream.str(std::string());
  sstream << "] (" << ID_LENGTH << "=" << totalLength;
  output += sstream.str();
  output += ")\n";
}

template <typename T>
void PrintArrays(std::string& output, const SubArrays<T>& subArrays, const PrintArgs<T>& args = PrintArgs<T>())
{
  T zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;
  size_t sectionLength = args.sectionLength;

  uintt count = subArrays.size();
  T** arrays = new T* [count];
  uintt* lengths = new uintt[count];

  for (uintt fa = 0; fa < count; ++fa)
  {
    arrays[fa] = subArrays[fa].first;
    lengths[fa] = subArrays[fa].second;
  }

  PrintArrays(output, arrays, lengths, count, PrintArgs<T>(zeroLimit, repeats, pipe, endl, sectionLength));
  delete[] arrays;
  delete[] lengths;
}

template <typename T>
void PrintArray(std::string& output, T* array, uintt length, const PrintArgs<T>& args = PrintArgs<T>())
{
  T zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;
  size_t sectionLength = args.sectionLength;

  T* arrays[] = {array};
  uintt lengths[] = {length};
  PrintArrays(output, arrays, lengths, 1, PrintArgs<T>(zeroLimit, repeats, pipe, endl, sectionLength));
}

template <typename T>
void PrintReValues(std::string& output, const MatrixRange& matrixRange, const PrintArgs<T>& args = PrintArgs<T>())
{
  T zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;
  size_t sectionLength = args.sectionLength;

  SubArrays<floatt> subArrrays;
  matrixRange.getReSubArrays(subArrrays);
  PrintArrays(output, subArrrays, PrintArgs<T>(zeroLimit, repeats, pipe, endl, sectionLength));
}

template <typename T>
void PrintImValues(std::string& output, const MatrixRange& matrixRange, const PrintArgs<T>& args = PrintArgs<T>())
{
  T zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;
  size_t sectionLength = args.sectionLength;

  SubArrays<floatt> subArrrays;
  matrixRange.getImSubArrays(subArrrays);
  PrintArrays(output, subArrrays, PrintArgs<T>(zeroLimit, repeats, pipe, endl, sectionLength));
}

inline void PrintMatrix(std::string& output, const MatrixRange& matrixRange, const PrintArgs<floatt>& args = PrintArgs<floatt>())
{
  floatt zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;

  std::stringstream sstream;
  sstream << "(" << ID_COLUMNS << "=" << matrixRange.getColumns() << ", " << ID_ROWS << "=" << matrixRange.getRows() << ") ";
  output += sstream.str();
  std::string output1;

  size_t sectionLength = matrixRange.getColumns() > 1 ? matrixRange.getColumns() : matrixRange.getRows();

  if (matrixRange.isReValues())
  {
    PrintReValues(output1, matrixRange, PrintArgs<floatt>(zeroLimit, repeats, pipe, endl, sectionLength));
    output += output1 + " ";
  }
  if (matrixRange.isImValues())
  {
    PrintImValues(output1, matrixRange, PrintArgs<floatt>(zeroLimit, repeats, pipe, endl, sectionLength));
    output += output1;
  }
}

inline void PrintMatrix(std::string& output, const math::Matrix* matrix, const PrintArgs<floatt>& args = PrintArgs<floatt>())
{
  floatt zeroLimit = args.zeroLimit;
  bool repeats = args.repeats;
  bool pipe = args.pipe;
  bool endl = args.endl;

  if (matrix == NULL)
  {
    output = "nullptr";
    return;
  }
  MatrixRange matrixRange(matrix);
  PrintMatrix(output, matrixRange, PrintArgs<floatt>(zeroLimit, repeats, pipe, endl));
}

}

#endif
