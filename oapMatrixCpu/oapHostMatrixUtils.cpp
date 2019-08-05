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

#include <algorithm>
#include <cstring>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <sstream>
#include <vector>

#include <linux/fs.h>

#include "oapHostMatrixUtils.h"

#include "oapHostMatrixUPtr.h"

#include "MatrixParser.h"
#include "MatrixPrinter.h"
#include "ReferencesCounter.h"

#include "MatricesList.h"

#define ReIsNotNULL(m) m->reValues != nullptr
#define ImIsNotNULL(m) m->imValues != nullptr

#ifdef DEBUG

std::ostream& operator<<(std::ostream& output, const math::Matrix*& matrix)
{
  return output << matrix << ", [" << matrix->columns << ", " << matrix->rows
         << "]";
}

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#else

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillWithValue (floatt* values, floatt value, uintt length)
{
  math::Memset(values, value, length);
}

inline void fillRePart(math::Matrix* output, floatt value)
{
  fillWithValue (output->reValues, value, output->columns * output->rows);
}

inline void fillImPart(math::Matrix* output, floatt value)
{
  fillWithValue (output->imValues, value, output->columns * output->rows);
}

namespace oap
{
namespace host
{

namespace
{
MatricesList gMatricesList ("HOST");
}

math::Matrix* NewMatrixRef (const math::Matrix* matrix, floatt value)
{
  math::Matrix* output = nullptr;
  if (matrix->reValues != nullptr && matrix->imValues != nullptr)
  {
    output = NewMatrix(matrix->columns, matrix->rows, value);
  }
  else if (matrix->reValues != nullptr)
  {
    output = NewReMatrix(matrix->columns, matrix->rows, value);
  }
  else if (matrix->imValues != nullptr)
  {
    output = NewImMatrix(matrix->columns, matrix->rows, value);
  }
  return output;
}

math::Matrix* NewMatrix(const math::Matrix* matrix, uintt columns, uintt rows, floatt value)
{
  math::Matrix* output = nullptr;
  if (matrix->reValues != nullptr && matrix->imValues != nullptr)
  {
    output = NewMatrix(columns, rows, value);
  }
  else if (matrix->reValues != nullptr)
  {
    output = NewReMatrix(columns, rows, value);
  }
  else if (matrix->imValues != nullptr)
  {
    output = NewImMatrix(columns, rows, value);
  }
  return output;
}

math::Matrix* NewMatrix(const math::MatrixInfo& matrixInfo, floatt value)
{
  return NewMatrix(matrixInfo.isRe, matrixInfo.isIm,
                   matrixInfo.m_matrixDim.columns, matrixInfo.m_matrixDim.rows,
                   value);
}

math::Matrix* NewMatrix(bool isre, bool isim, uintt columns, uintt rows,
                        floatt value)
{
  if (isre && isim)
  {
    return oap::host::NewMatrix(columns, rows, value);
  }
  else if (isre)
  {
    return oap::host::NewReMatrix(columns, rows, value);
  }
  else if (isim)
  {
    return oap::host::NewImMatrix(columns, rows, value);
  }
  return nullptr;
}

math::Matrix* allocMatrix (bool isRe, bool isIm, uintt columns, uintt rows, floatt value, floatt* rebuffer = nullptr, floatt* imbuffer = nullptr)
{
  math::Matrix* output = NEW_MATRIX();
  uintt length = columns * rows;

  output->realColumns = columns;
  output->columns = columns;
  output->realRows = rows;
  output->rows = rows;

  auto set = [length, value] (floatt** output, bool is, floatt* buffer)
  {
    floatt* tmp = nullptr;
    if (is)
    {
      if (buffer)
      {
        tmp = buffer;
      }
      else
      {
        tmp = new floatt [length];
        fillWithValue (tmp, value, length);
      }
    }
    *output = tmp;
  };

  set (&(output->reValues), isRe, rebuffer);
  set (&(output->imValues), isIm, imbuffer);

  gMatricesList.add (output, CreateMatrixInfo (output));

  return output;
}

math::Matrix* NewMatrix(uintt columns, uintt rows, floatt value)
{
  return allocMatrix (true, true, columns, rows, value);
}

math::Matrix* NewReMatrix(uintt columns, uintt rows, floatt value)
{
  return allocMatrix (true, false, columns, rows, value);
}

math::Matrix* NewImMatrix(uintt columns, uintt rows, floatt value)
{
  return allocMatrix (false, true, columns, rows, value);
}

math::Matrix* NewMatrix(const std::string& text)
{
  matrixUtils::Parser parser(text);

  uintt columns = 0;
  uintt rows = 0;

  bool iscolumns = false;
  bool isrows = false;

  if (parser.getColumns(columns) == true)
  {
    iscolumns = true;
  }
  if (parser.getRows(rows) == true)
  {
    isrows = true;
  }
  std::pair<floatt*, size_t> pairRe = matrixUtils::CreateArray(text, 1);
  std::pair<floatt*, size_t> pairIm = matrixUtils::CreateArray(text, 2);

  debugAssert(pairRe.first == nullptr || pairIm.first == nullptr ||
              pairRe.second == pairIm.second);

  floatt* revalues = pairRe.first;
  floatt* imvalues = pairIm.first;

  if ( (iscolumns && isrows) == false)
  {
    size_t sq = sqrt(pairRe.second);
    columns = sq;
    rows = sq;
    iscolumns = true;
    isrows = true;
  }
  else if (iscolumns && !isrows)
  {
    rows = pairRe.second / columns;
    isrows = true;
  }
  else if (isrows && !iscolumns)
  {
    columns = pairRe.second / rows;
    iscolumns = true;
  }

  if (revalues == nullptr && imvalues == nullptr)
  {
    return nullptr;
  }

  return allocMatrix (revalues != nullptr, imvalues != nullptr, columns, rows, 0, revalues, imvalues);
}

void DeleteMatrix(const math::Matrix* matrix)
{
  if (nullptr == matrix)
  {
    return;
  }

  auto minfo = gMatricesList.remove (matrix);

  if (matrix->reValues != nullptr)
  {
    delete[] matrix->reValues;
  }
  if (matrix->imValues != nullptr)
  {
    delete[] matrix->imValues;
  }

  DELETE_MATRIX(matrix);

  if (minfo.isInitialized ())
  {
    logTrace ("Deallocate: host matrix = %p %s", matrix, minfo.toString().c_str());
  }
}

floatt GetReValue(const math::Matrix* matrix, uintt column, uintt row)
{
  if (matrix->reValues == nullptr)
  {
    return 0;
  }
  return matrix->reValues[row * matrix->columns + column];
}

floatt GetImValue(const math::Matrix* matrix, uintt column, uintt row)
{
  if (matrix->imValues == nullptr)
  {
    return 0;
  }
  return matrix->imValues[row * matrix->columns + column];
}

void SetReValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value)
{
  if (matrix->reValues)
  {
    matrix->reValues[row * matrix->columns + column] = value;
  }
}

void SetImValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value)
{
  if (matrix->imValues)
  {
    matrix->imValues[row * matrix->columns + column] = value;
  }
}

std::string GetMatrixStr(const math::Matrix* matrix)
{
  std::string output;
  matrixUtils::PrintMatrix (output, matrix, matrixUtils::PrintArgs());
  return output;
}

void PrintMatrix(FILE* stream, const matrixUtils::PrintArgs& args, const math::Matrix* matrix)
{
  std::string output;
  matrixUtils::PrintMatrix (output, matrix, args);
  fprintf(stream, "%s", output.c_str());
}

void PrintMatrix(FILE* stream, const math::Matrix* matrix, const matrixUtils::PrintArgs& args)
{
  PrintMatrix(stream, args, matrix);
}

void PrintMatrix(const matrixUtils::PrintArgs& args, const math::Matrix* matrix)
{
  PrintMatrix(stdout, args, matrix);
}

void PrintMatrix(const math::Matrix* matrix, const matrixUtils::PrintArgs& args)
{
  PrintMatrix(args, matrix);
}

bool PrintMatrixToFile(const std::string& path, const matrixUtils::PrintArgs& args, const math::Matrix* matrix)
{
  FILE* file = fopen(path.c_str(), "w");

  if (file == nullptr)
  {
    return false;
  }

  PrintMatrix (file, args, matrix);

  fclose(file);
  return true;
}

bool PrintMatrixToFile(const std::string& path, const math::Matrix* matrix, const matrixUtils::PrintArgs& args)
{
  return PrintMatrixToFile(path, args, matrix);
}

void Copy(math::Matrix* dst, const math::Matrix* src, const MatrixEx& subMatrix,
          uintt column, uintt row)
{
  uintt rows = dst->rows;
  uintt columns2 = subMatrix.dims.columns;
  for (uintt fa = 0; fa < rows; fa++)
  {
    uintt fa1 = fa + subMatrix.row;
    if (fa < row)
    {
      Copy(dst->reValues + fa * dst->columns,
           src->reValues + (fa1)*columns2, column);
      Copy(dst->reValues + column + fa * dst->columns,
           src->reValues + (1 + column) + fa * columns2,
           (columns2 - column));
    }
    else if (fa >= row)
    {
      Copy(dst->reValues + fa * dst->columns,
           &src->reValues[(fa1 + 1) * columns2], column);

      Copy(dst->reValues + column + fa * dst->columns,
           &src->reValues[(fa1 + 1) * columns2 + column + 1],
           (columns2 - column));
    }
  }
}

void Copy(math::Matrix* dst, const math::Matrix* src, uintt column, uintt row)
{
  uintt rows = src->rows;
  uintt columns = src->columns;
  for (uintt fa = 0; fa < rows; fa++)
  {
    if (fa < row)
    {
      Copy(&dst->reValues[fa * dst->columns],
           &src->reValues[fa * columns], column);
      if (column < src->columns - 1)
      {
        Copy(&dst->reValues[column + fa * dst->columns],
             &src->reValues[(1 + column) + fa * columns],
             (src->columns - (column + 1)));
      }
    }
    else if (fa > row)
    {
      Copy(&dst->reValues[(fa - 1) * dst->columns],
           &src->reValues[fa * columns], column);
      if (column < src->columns - 1)
      {
        Copy(&dst->reValues[column + (fa - 1) * dst->columns],
             &src->reValues[fa * columns + (column + 1)],
             (src->columns - (column + 1)));
      }
    }
  }
}

void CopyMatrix(math::Matrix* dst, const math::Matrix* src)
{
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  if (length1 == length2)
  {
    if (ReIsNotNULL(dst) && ReIsNotNULL(src))
    {
      CopyBuffer(dst->reValues, src->reValues, length1);
    }
    if (ImIsNotNULL(dst) && ImIsNotNULL(src))
    {
      CopyBuffer(dst->imValues, src->imValues, length1);
    }
  }
  else if (length1 < length2 && dst->columns <= src->columns &&
           dst->rows <= src->rows)
  {
    CopySubMatrix(dst, src, 0, 0);
  }
}

void CopyRe(math::Matrix* dst, const math::Matrix* src)
{
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  if (ReIsNotNULL(dst) && ReIsNotNULL(src))
  {
    memcpy(dst->reValues, src->reValues, length * sizeof(floatt));
  }
}

void CopyIm(math::Matrix* dst, const math::Matrix* src)
{
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  if (ImIsNotNULL(dst) && ImIsNotNULL(src))
  {
    memcpy(dst->imValues, src->imValues, length * sizeof(floatt));
  }
}

math::Matrix* NewMatrixCopy(const math::Matrix* matrix)
{
  math::Matrix* output = oap::host::NewMatrixRef (matrix);
  oap::host::CopyMatrix(output, matrix);
  return output;
}

void SetVector(math::Matrix* matrix, uintt column, math::Matrix* vector)
{
  SetVector(matrix, column, vector->reValues, vector->imValues, vector->rows);
}

void SetVector(math::Matrix* matrix, uintt column, floatt* revector,
               floatt* imvector, uintt length)
{
  if (revector != nullptr)
  {
    SetReVector(matrix, column, revector, length);
  }

  if (imvector != nullptr)
  {
    SetImVector(matrix, column, imvector, length);
  }
}

void SetReVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length)
{
  if (matrix->reValues)
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      matrix->reValues[column + matrix->columns * fa] = vector[fa];
    }
  }
}

void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length)
{
  if (matrix->reValues)
  {
    memcpy(&matrix->reValues[row * matrix->columns], vector,
           length * sizeof(floatt));
  }
}

void SetImVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length)
{
  if (matrix->imValues)
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      matrix->imValues[column + matrix->columns * fa] = vector[fa];
    }
  }
}

void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length)
{
  if (matrix->imValues)
  {
    memcpy(&matrix->imValues[row * matrix->columns], vector,
           length * sizeof(floatt));
  }
}

void SetReVector(math::Matrix* matrix, uintt column, floatt* vector)
{
  SetReVector(matrix, column, vector, matrix->rows);
}

void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector)
{
  SetTransposeReVector(matrix, row, vector, matrix->columns);
}

void SetImVector(math::Matrix* matrix, uintt column, floatt* vector)
{
  SetImVector(matrix, column, vector, matrix->rows);
}

void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector)
{
  SetTransposeImVector(matrix, row, vector, matrix->columns);
}

void GetMatrixStr(std::string& text, const math::Matrix* matrix)
{
  matrixUtils::PrintMatrix(text, matrix, matrixUtils::PrintArgs());
}

void ToString (std::string& str, const math::Matrix* matrix)
{
  if (matrix == nullptr)
  {
    str = "nullptr";
    return;
  }
  matrixUtils::PrintMatrix(str, matrix, matrixUtils::PrintArgs());
}

void GetReMatrixStr(std::string& text, const math::Matrix* matrix)
{
  matrixUtils::PrintMatrix(text, matrix, matrixUtils::PrintArgs());
}

void GetImMatrixStr(std::string& str, const math::Matrix* matrix)
{
  str = "";
  if (matrix == nullptr)
  {
    return;
  }
  std::stringstream sstream;
  str += "[";
  for (int fb = 0; fb < matrix->rows; fb++)
  {
    for (int fa = 0; fa < matrix->columns; fa++)
    {
      sstream << matrix->imValues[fb * matrix->columns + fa];
      str += sstream.str();
      sstream.str("");
      if (fa != matrix->columns - 1)
      {
        str += ",";
      }
      if (fa == matrix->columns - 1 && fb != matrix->rows - 1)
      {
        str += "\n";
      }
    }
  }
  str += "]";
}

void GetVector(math::Matrix* vector, math::Matrix* matrix, uintt column)
{
  GetVector(vector->reValues, vector->imValues, vector->rows, matrix, column);
}

void GetVector(floatt* revector, floatt* imvector, uint length, math::Matrix* matrix, uint column)
{
  if (revector != nullptr)
  {
    GetReVector(revector, length, matrix, column);
  }

  if (imvector != nullptr)
  {
    GetImVector(imvector, length, matrix, column);
  }
}

void GetTransposeVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  if (vector->reValues != nullptr)
  {
    GetTransposeReVector(vector, matrix, column);
  }
  if (vector->imValues != nullptr)
  {
    GetTransposeImVector(vector, matrix, column);
  }
}

void GetTransposeReVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  GetTransposeReVector(vector->reValues, matrix, column);
}

void GetTransposeImVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  GetTransposeImVector(vector->imValues, matrix, column);
}

void GetReVector(floatt* vector, uint length, math::Matrix* matrix, uint column)
{
  if (matrix->reValues)
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      vector[fa] = matrix->reValues[column + matrix->columns * fa];
    }
  }
}

void GetTransposeReVector(floatt* vector, uint length, math::Matrix* matrix, uint row)
{
  if (matrix->reValues)
  {
    memcpy(vector, &matrix->reValues[row * matrix->columns],
           length * sizeof(floatt));
  }
}

void GetImVector(floatt* vector, uint length, math::Matrix* matrix, uint column)
{
  if (matrix->imValues)
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      vector[fa] = matrix->imValues[column + matrix->columns * fa];
    }
  }
}

void GetTransposeImVector(floatt* vector, uint length, math::Matrix* matrix, uint row)
{
  if (matrix->imValues)
  {
    memcpy(vector, &matrix->imValues[row * matrix->columns],
           length * sizeof(floatt));
  }
}

void GetReVector(floatt* vector, math::Matrix* matrix, uint column)
{
  GetReVector(vector, matrix->rows, matrix, column);
}

void GetTransposeReVector(floatt* vector, math::Matrix* matrix, uint row)
{
  GetTransposeReVector(vector, matrix->columns, matrix, row);
}

void GetImVector(floatt* vector, math::Matrix* matrix, uint column)
{
  GetImVector(vector, matrix->rows, matrix, column);
}

void GetTransposeImVector(floatt* vector, math::Matrix* matrix, uint row)
{
  GetTransposeReVector(vector, matrix->columns, matrix, row);
}

floatt SmallestDiff(math::Matrix* matrix, math::Matrix* matrix1)
{
  floatt diff = matrix->reValues[0] - matrix1->reValues[0];
  for (uintt fa = 0; fa < matrix->columns; fa++)
  {
    for (uintt fb = 0; fb < matrix->rows; fb++)
    {
      uintt index = fa + fb * matrix->columns;
      floatt diff1 = matrix->reValues[index] - matrix1->reValues[index];
      if (diff1 < 0)
      {
        diff1 = -diff1;
      }
      if (diff > diff1)
      {
        diff = diff1;
      }
    }
  }
  return diff;
}

floatt LargestDiff(math::Matrix* matrix, math::Matrix* matrix1)
{
  floatt diff = matrix->reValues[0] - matrix1->reValues[0];
  for (uintt fa = 0; fa < matrix->columns; fa++)
  {
    for (uintt fb = 0; fb < matrix->rows; fb++)
    {
      uintt index = fa + fb * matrix->columns;
      floatt diff1 = matrix->reValues[index] - matrix1->reValues[index];
      if (diff1 < 0)
      {
        diff1 = -diff1;
      }
      if (diff < diff1)
      {
        diff = diff1;
      }
    }
  }
  return diff;
}

void SetIdentity(math::Matrix* matrix)
{
  oap::host::SetDiagonalReMatrix(matrix, 1);
  oap::host::SetImZero(matrix);
}

void SetReZero(math::Matrix* matrix)
{
  if (matrix->reValues)
  {
    memset(matrix->reValues, 0,
           matrix->columns * matrix->rows * sizeof(floatt));
  }
}

void SetImZero(math::Matrix* matrix)
{
  if (matrix->imValues)
  {
    memset(matrix->imValues, 0,
           matrix->columns * matrix->rows * sizeof(floatt));
  }
}

void SetZero(math::Matrix* matrix)
{
  SetReZero(matrix);
  SetImZero(matrix);
}

void SetIdentityMatrix(math::Matrix* matrix)
{
  SetDiagonalReMatrix(matrix, 1);
  SetImZero(matrix);
}

bool IsEquals(math::Matrix* transferMatrix2, math::Matrix* transferMatrix1,
              floatt diff)
{
  for (uintt fa = 0; fa < transferMatrix2->columns; fa++)
  {
    for (uintt fb = 0; fb < transferMatrix2->rows; fb++)
    {
      floatt p = transferMatrix2->reValues[fa + transferMatrix2->columns * fb] -
                 transferMatrix1->reValues[fa + transferMatrix1->columns * fb];
      if (p < -diff || p > diff)
      {
        return false;
      }
    }
  }
  return true;
}

floatt GetTrace(math::Matrix* matrix)
{
  floatt o = 1.;
  for (uintt fa = 0; fa < matrix->columns; ++fa)
  {
    floatt v = matrix->reValues[fa * matrix->columns + fa];
    if (-MATH_VALUE_LIMIT < v && v < MATH_VALUE_LIMIT)
    {
      v = 0;
    }
    o = o * v;
  }
  return o;
}

void SetDiagonalMatrix(math::Matrix* matrix, floatt a)
{
  SetDiagonalReMatrix(matrix, a);
  SetDiagonalImMatrix(matrix, a);
}

void SetDiagonalReMatrix(math::Matrix* matrix, floatt a)
{
  if (matrix->reValues)
  {
    fillRePart(matrix, 0);
    for (int fa = 0; fa < matrix->columns; fa++)
    {
      matrix->reValues[fa * matrix->columns + fa] = a;
    }
  }
}

void SetDiagonalImMatrix(math::Matrix* matrix, floatt a)
{
  if (matrix->imValues)
  {
    fillImPart(matrix, 0);
    for (int fa = 0; fa < matrix->columns; fa++)
    {
      matrix->imValues[fa * matrix->columns + fa] = a;
    }
  }
}

math::MatrixInfo CreateMatrixInfo(const math::Matrix* matrix)
{
  return math::MatrixInfo (matrix->reValues != nullptr,
                           matrix->imValues != nullptr,
                           matrix->columns,
                           matrix->rows);
}

math::MatrixInfo GetMatrixInfo (const math::Matrix* matrix)
{
  return gMatricesList.getMatrixInfo (matrix);
}

void SetSubs(math::Matrix* matrix, uintt subcolumns, uintt subrows)
{
  SetSubColumns(matrix, subcolumns);
  SetSubRows(matrix, subrows);
}

void SetSubColumns(math::Matrix* matrix, uintt subcolumns)
{
  if (subcolumns == MATH_UNDEFINED)
  {
    matrix->columns = matrix->realColumns;
  }
  else
  {
    matrix->columns = subcolumns;
    debugAssert(matrix->columns <= matrix->realColumns);
  }
}

void SetSubRows(math::Matrix* matrix, uintt subrows)
{
  if (subrows == MATH_UNDEFINED)
  {
    matrix->rows = matrix->realRows;
  }
  else
  {
    matrix->rows = subrows;
    debugAssert(matrix->rows <= matrix->realRows);
  }
}

void SetSubsSafe(math::Matrix* matrix, uintt subcolumns, uintt subrows)
{
  SetSubColumnsSafe(matrix, subcolumns);
  SetSubRowsSafe(matrix, subrows);
}

void SetSubColumnsSafe(math::Matrix* matrix, uintt subcolumns)
{
  if (subcolumns == MATH_UNDEFINED || matrix->columns < subcolumns)
  {
    matrix->columns = matrix->realColumns;
  }
  else
  {
    matrix->columns = subcolumns;
  }
}

void SetSubRowsSafe(math::Matrix* matrix, uintt subrows)
{
  if (subrows == MATH_UNDEFINED || matrix->rows < subrows)
  {
    matrix->rows = matrix->realRows;
  }
  else
  {
    debugAssert(matrix->rows <= matrix->realRows);
  }
}

math::Matrix* ReadMatrix (const std::string& path)
{
  utils::ByteBuffer buffer (path);
  math::Matrix* matrix = oap::host::LoadMatrix (buffer);

  return matrix;
}

math::Matrix* ReadRowVector (const std::string& path, size_t index)
{
  oap::HostMatrixUPtr matrix = ReadMatrix (path);
  math::Matrix* subMatrix = oap::host::NewSubMatrix (matrix, 0, index, matrix->columns, 1);
  return subMatrix;
}

math::Matrix* ReadColumnVector (const std::string& path, size_t index)
{
  oap::HostMatrixUPtr matrix = ReadMatrix (path);
  math::Matrix* subMatrix = oap::host::NewSubMatrix (matrix, index, 0, 1, matrix->rows);
  return subMatrix;
}

void CopyReBuffer (math::Matrix* houtput, math::Matrix* hinput)
{
  size_t sOutput = houtput->columns * houtput->rows;
  size_t sInput = hinput->columns * hinput->rows;

  debugExceptionMsg(sOutput == sInput, "Buffers have different sizes.");

  memcpy (houtput->reValues, hinput->reValues, sOutput * sizeof (floatt));
}

bool WriteMatrix (const std::string& path, const math::Matrix* matrix)
{
  utils::ByteBuffer buffer;
  oap::host::SaveMatrix (matrix, buffer);
  try
  {
    buffer.fwrite (path);
  }
  catch (const std::runtime_error& error)
  {
    debugError ("Write to file error: %s", error.what());
    return false;
  }
  return true;
}

void copySubMatrix (math::Matrix* dst, const math::Matrix* src, uintt cindex, uintt rindex)
{
  if (ReIsNotNULL(dst) && ReIsNotNULL(src))
  {
    for (uintt fa = 0; fa < dst->rows; ++fa)
    {
      CopyBuffer(GetRePtr(dst, 0, fa), GetRePtr(src, cindex, fa + rindex), dst->columns);
    }
  }
  if (ImIsNotNULL(dst) && ImIsNotNULL(src))
  {
    for (uintt fa = 0; fa < dst->rows; ++fa)
    {
      CopyBuffer(GetImPtr(dst, 0, fa), GetImPtr(src, cindex, fa + rindex), dst->columns);
    }
  }
}

void CopySubMatrix(math::Matrix* dst, const math::Matrix* src, uintt cindex, uintt rindex)
{
  copySubMatrix (dst, src, cindex, rindex);
}

inline uintt calculate (uintt matrixd, uintt dindex, uintt dlength)
{
  return dindex + dlength < matrixd ? dlength : matrixd - dindex;
}

math::Matrix* NewSubMatrix (const math::Matrix* orig, uintt cindex, uintt rindex, uintt clength, uintt rlength)
{
  clength = calculate (orig->columns, cindex, clength);
  rlength = calculate (orig->rows, rindex, rlength);

  math::Matrix* submatrix = oap::host::NewMatrix (orig, clength, rlength);
  copySubMatrix (submatrix, orig, cindex, rindex);
  return submatrix;
}

math::Matrix* GetSubMatrix (const math::Matrix* orig, uintt cindex, uintt rindex, math::Matrix* matrix)
{
  uintt clength = calculate (orig->columns, cindex, matrix->columns);
  uintt rlength = calculate (orig->rows, rindex, matrix->rows);

  if (matrix->columns == clength && matrix->rows == rlength)
  {
    copySubMatrix (matrix, orig, cindex, rindex);
    return matrix;
  }

  oap::host::DeleteMatrix (matrix);
  return NewSubMatrix (orig, cindex, rindex, clength, rlength);
}

void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer)
{
  buffer.push_back (minfo.isRe);
  buffer.push_back (minfo.isIm);
  buffer.push_back (minfo.columns ());
  buffer.push_back (minfo.rows ());
}

void SaveMatrix (const math::Matrix* matrix, utils::ByteBuffer& buffer)
{
  bool isMatrix = (matrix != nullptr);

  buffer.push_back (isMatrix);

  if (!isMatrix)
  {
    return;
  }

  auto minfo = oap::host::GetMatrixInfo (matrix);

  SaveMatrixInfo (minfo, buffer);

  if (minfo.isRe)
  {
    buffer.push_back (matrix->reValues, minfo.length ());
  }

  if (minfo.isIm)
  {
    buffer.push_back (matrix->imValues, minfo.length ());
  }
}

math::Matrix* LoadMatrix (const utils::ByteBuffer& buffer)
{
  bool isMatrix = buffer.read <bool>();

  if (!isMatrix)
  {
    return nullptr;
  }

  math::MatrixInfo minfo = LoadMatrixInfo (buffer);
  math::Matrix* matrix = NewMatrix (minfo);

  if (minfo.isRe)
  {
    buffer.read (matrix->reValues, minfo.length ());
  }

  if (minfo.isIm)
  {
    buffer.read (matrix->imValues, minfo.length ());
  }

  return matrix;
}

math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer)
{
  math::MatrixInfo minfo;

  minfo.isRe = buffer.read<decltype (minfo.isRe)> ();
  minfo.isIm = buffer.read<decltype (minfo.isIm)> ();
  minfo.m_matrixDim.columns = buffer.read<decltype (minfo.m_matrixDim.columns)> ();
  minfo.m_matrixDim.rows = buffer.read<decltype (minfo.m_matrixDim.rows)> ();

  return minfo;
}

void CopyArrayToMatrix (math::Matrix* matrix, void* rebuffer, void* imbuffer)
{
  if (rebuffer != nullptr)
  {
    CopyArrayToReMatrix (matrix, rebuffer);
  }
  if (imbuffer != nullptr)
  {
    CopyArrayToImMatrix (matrix, imbuffer);
  }
}

void CopyArrayToReMatrix (math::Matrix* matrix, void* buffer)
{
  debugAssert (matrix->reValues == nullptr);
  memcpy (matrix->reValues, buffer, matrix->columns * matrix->rows);
}

void CopyArrayToImMatrix (math::Matrix* matrix, void* buffer)
{
  debugAssert (matrix->imValues == nullptr);
  memcpy (matrix->reValues, buffer, matrix->columns * matrix->rows);
}

}
}
