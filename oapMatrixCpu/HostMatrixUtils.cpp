/*
 * Copyright 2016, 2017 Marcin Matula
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

#include "HostMatrixUtils.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <memory>
#include <linux/fs.h>
#include "MatrixUtils.h"
#include "ArrayTools.h"
#include "ReferencesCounter.h"

#define ReIsNotNULL(m) m->reValues != NULL
#define ImIsNotNULL(m) m->imValues != NULL

#ifdef DEBUG

std::ostream& operator<<(std::ostream& output, const math::Matrix*& matrix) {
  return output << matrix << ", [" << matrix->columns << ", " << matrix->rows
                << "]";
}

class MatricesCounter : public ReferencesCounter<math::Matrix*> {
 public:
  MatricesCounter() : ReferencesCounter<math::Matrix*>("MatricesCounter") {}

  static MatricesCounter& GetInstance();

 protected:
  virtual math::Matrix* createObject() { return new math::Matrix(); }

  virtual void destroyObject(math::Matrix*& t) { delete t; }

 private:
  static MatricesCounter m_matricesCounter;
};

MatricesCounter MatricesCounter::m_matricesCounter;

MatricesCounter& MatricesCounter::GetInstance() {
  return MatricesCounter::m_matricesCounter;
}

#define NEW_MATRIX() MatricesCounter::GetInstance().create();

#define DELETE_MATRIX(matrix) MatricesCounter::GetInstance().destroy(matrix);

#else

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillRePart(math::Matrix* output, floatt value) {
  math::Memset(output->reValues, value, output->columns * output->rows);
}

inline void fillImPart(math::Matrix* output, floatt value) {
  math::Memset(output->imValues, value, output->columns * output->rows);
}

namespace host {

math::Matrix* NewMatrix(const math::Matrix* matrix, floatt value) {
  math::Matrix* output = NULL;
  if (matrix->reValues != NULL && matrix->imValues != NULL) {
    output = NewMatrix(matrix->columns, matrix->rows, value);
  } else if (matrix->reValues != NULL) {
    output = NewReMatrix(matrix->columns, matrix->rows, value);
  } else if (matrix->imValues != NULL) {
    output = NewImMatrix(matrix->columns, matrix->rows, value);
  }
  return output;
}

math::Matrix* NewMatrix(const math::Matrix* matrix, uintt columns, uintt rows, floatt value) {
  math::Matrix* output = NULL;
  if (matrix->reValues != NULL && matrix->imValues != NULL) {
    output = NewMatrix(columns, rows, value);
  } else if (matrix->reValues != NULL) {
    output = NewReMatrix(columns, rows, value);
  } else if (matrix->imValues != NULL) {
    output = NewImMatrix(columns, rows, value);
  }
  return output;
}

math::Matrix* NewMatrix(const math::MatrixInfo& matrixInfo, floatt value) {
  return NewMatrix(matrixInfo.isRe, matrixInfo.isIm,
                   matrixInfo.m_matrixDim.columns, matrixInfo.m_matrixDim.rows,
                   value);
}

math::Matrix* NewMatrix(bool isre, bool isim, uintt columns, uintt rows,
                        floatt value) {
  if (isre && isim) {
    return host::NewMatrix(columns, rows, value);
  } else if (isre) {
    return host::NewReMatrix(columns, rows, value);
  } else if (isim) {
    return host::NewImMatrix(columns, rows, value);
  }
  return nullptr;
}

math::Matrix* NewMatrix(uintt columns, uintt rows, floatt value) {
  math::Matrix* output = NEW_MATRIX();
  uintt length = columns * rows;

  output->realColumns = columns;
  output->columns = columns;
  output->realRows = rows;
  output->rows = rows;

  output->reValues = new floatt[length];
  output->imValues = new floatt[length];

  fillRePart(output, value);
  fillImPart(output, value);
  return output;
}

math::Matrix* NewReMatrix(uintt columns, uintt rows, floatt value) {
  math::Matrix* output = NEW_MATRIX();
  uintt length = columns * rows;

  output->realColumns = columns;
  output->columns = columns;
  output->realRows = rows;
  output->rows = rows;

  output->reValues = new floatt[length];
  output->imValues = NULL;

  fillRePart(output, value);
  return output;
}

math::Matrix* NewImMatrix(uintt columns, uintt rows, floatt value) {
  math::Matrix* output = NEW_MATRIX();
  uintt length = columns * rows;

  output->realColumns = columns;
  output->columns = columns;
  output->realRows = rows;
  output->rows = rows;

  output->reValues = NULL;
  output->imValues = new floatt[length];

  fillImPart(output, value);
  return output;
}

math::Matrix* NewMatrix(const std::string& text) {
  matrixUtils::Parser parser(text);

  uintt columns = 0;
  uintt rows = 0;

  bool iscolumns = false;
  bool isrows = false;

  if (parser.getColumns(columns) == true) {
    iscolumns = true;
  }
  if (parser.getRows(rows) == true) {
    isrows = true;
  }
  std::pair<floatt*, size_t> pairRe = matrixUtils::CreateArray(text, 1);
  std::pair<floatt*, size_t> pairIm = matrixUtils::CreateArray(text, 2);

  debugAssert(pairRe.first == NULL || pairIm.first == NULL ||
              pairRe.second == pairIm.second);

  floatt* revalues = pairRe.first;
  floatt* imvalues = pairIm.first;

  if ( (iscolumns && isrows) == false) {
    size_t sq = sqrt(pairRe.second);
    columns = sq;
    rows = sq;
    iscolumns = true;
    isrows = true;
  } else if (iscolumns && !isrows) {
    rows = pairRe.second / columns;
    isrows = true;
  } else if (isrows && !iscolumns) {
    columns = pairRe.second / rows;
    iscolumns = true;
  }

  if (revalues == NULL && imvalues == NULL) {
    return NULL;
  }

  math::Matrix* matrix = NEW_MATRIX();
  matrix->columns = columns;
  matrix->realColumns = columns;
  matrix->rows = rows;
  matrix->realRows = rows;
  matrix->reValues = revalues;
  matrix->imValues = imvalues;

  return matrix;
}

void DeleteMatrix(math::Matrix* matrix) {
  if (NULL == matrix) {
    return;
  }
  if (matrix->reValues != NULL) {
    delete[] matrix->reValues;
  }
  if (matrix->imValues != NULL) {
    delete[] matrix->imValues;
  }
  DELETE_MATRIX(matrix);
}

floatt GetReValue(const math::Matrix* matrix, uintt column, uintt row) {
  if (matrix->reValues == NULL) {
    return 0;
  }
  return matrix->reValues[row * matrix->columns + column];
}

floatt GetImValue(const math::Matrix* matrix, uintt column, uintt row) {
  if (matrix->imValues == NULL) {
    return 0;
  }
  return matrix->imValues[row * matrix->columns + column];
}

void SetReValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value) {
  if (matrix->reValues) {
    matrix->reValues[row * matrix->columns + column] = value;
  }
}

void SetImValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value) {
  if (matrix->imValues) {
    matrix->imValues[row * matrix->columns + column] = value;
  }
}

std::string GetMatrixStr(const math::Matrix* matrix) {
  std::string output;
  matrixUtils::PrintMatrix(output, matrix);
  return output;
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix) {
  PrintMatrix(stdout, text, matrix);
}

void PrintMatrix(FILE* stream, const std::string& text, const math::Matrix* matrix) {
  std::string output = text;
  if (text.size() > 0) {
    output += " ";
  }
  matrixUtils::PrintMatrix(output, matrix);
  fprintf(stream, "%s HOST \n", output.c_str());
}

void PrintMatrix(FILE* stream, const math::Matrix* matrix) {
  PrintMatrix(stdout, "", matrix);
}

void PrintMatrix(const math::Matrix* matrix) { PrintMatrix("", matrix); }

bool PrintMatrixToFile(const std::string& path, const std::string& text, const math::Matrix* matrix) {
  FILE* file = fopen(path.c_str(), "w");
  if (file == NULL) { return false; }
  PrintMatrix(file, text, matrix);
  fclose(file);
  return true;
}

bool PrintMatrixToFile(const std::string& path, const math::Matrix* matrix) {
  return PrintMatrixToFile(path, "", matrix);
}

void Copy(math::Matrix* dst, const math::Matrix* src, const MatrixEx& subMatrix,
          uintt column, uintt row) {
  uintt rows = dst->rows;
  uintt columns2 = subMatrix.columnsLength;
  for (uintt fa = 0; fa < rows; fa++) {
    uintt fa1 = fa + subMatrix.beginRow;
    if (fa < row) {
      Copy(dst->reValues + fa * dst->columns,
                   src->reValues + (fa1)*columns2, column);
      Copy(dst->reValues + column + fa * dst->columns,
                   src->reValues + (1 + column) + fa * columns2,
                   (columns2 - column));
    } else if (fa >= row) {
      Copy(dst->reValues + fa * dst->columns,
                   &src->reValues[(fa1 + 1) * columns2], column);

      Copy(dst->reValues + column + fa * dst->columns,
                   &src->reValues[(fa1 + 1) * columns2 + column + 1],
                   (columns2 - column));
    }
  }
}

void Copy(math::Matrix* dst, const math::Matrix* src, uintt column, uintt row) {
  uintt rows = src->rows;
  uintt columns = src->columns;
  for (uintt fa = 0; fa < rows; fa++) {
    if (fa < row) {
        Copy(&dst->reValues[fa * dst->columns],
                   &src->reValues[fa * columns], column);
      if (column < src->columns - 1) {
        Copy(&dst->reValues[column + fa * dst->columns],
                     &src->reValues[(1 + column) + fa * columns],
                     (src->columns - (column + 1)));
      }
    } else if (fa > row) {
      Copy(&dst->reValues[(fa - 1) * dst->columns],
                   &src->reValues[fa * columns], column);
      if (column < src->columns - 1) {
        Copy(&dst->reValues[column + (fa - 1) * dst->columns],
                     &src->reValues[fa * columns + (column + 1)],
                     (src->columns - (column + 1)));
      }
    }
  }
}

void CopySubMatrix(math::Matrix* dst, const math::Matrix* src);

void CopyMatrix(math::Matrix* dst, const math::Matrix* src) {
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  if (length1 == length2) {
    if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
      CopyBuffer(dst->reValues, src->reValues, length1);
    }
    if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
      CopyBuffer(dst->imValues, src->imValues, length1);
    }
  } else if (length1 < length2 && dst->columns <= src->columns &&
             dst->rows <= src->rows) {
    CopySubMatrix(dst, src);
  }
}

void CopySubMatrix(math::Matrix* dst, const math::Matrix* src) {
  if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
    for (uintt fa = 0; fa < dst->rows; ++fa) {
      CopyBuffer(GetRePtr(dst, 0, fa), GetRePtr(src, 0, fa), dst->columns);
    }
  }
  if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
    for (uintt fa = 0; fa < dst->rows; ++fa) {
      CopyBuffer(GetImPtr(dst, 0, fa), GetImPtr(src, 0, fa), dst->columns);
    }
  }
}

void CopyRe(math::Matrix* dst, const math::Matrix* src) {
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
    memcpy(dst->reValues, src->reValues, length * sizeof(floatt));
  }
}

void CopyIm(math::Matrix* dst, const math::Matrix* src) {
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
    memcpy(dst->imValues, src->imValues, length * sizeof(floatt));
  }
}

math::Matrix* NewMatrixCopy(const math::Matrix* matrix) {
  math::Matrix* output = host::NewMatrix(matrix);
  host::CopyMatrix(output, matrix);
  return output;
}

void SetVector(math::Matrix* matrix, uintt column, math::Matrix* vector) {
  SetVector(matrix, column, vector->reValues, vector->imValues, vector->rows);
}

void SetVector(math::Matrix* matrix, uintt column, floatt* revector,
               floatt* imvector, uintt length) {
  if (revector != NULL) {
    SetReVector(matrix, column, revector, length);
  }

  if (imvector != NULL) {
    SetImVector(matrix, column, imvector, length);
  }
}

void SetReVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length) {
  if (matrix->reValues) {
    for (uintt fa = 0; fa < length; fa++) {
      matrix->reValues[column + matrix->columns * fa] = vector[fa];
    }
  }
}

void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length) {
  if (matrix->reValues) {
    memcpy(&matrix->reValues[row * matrix->columns], vector,
           length * sizeof(floatt));
  }
}

void SetImVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length) {
  if (matrix->imValues) {
    for (uintt fa = 0; fa < length; fa++) {
      matrix->imValues[column + matrix->columns * fa] = vector[fa];
    }
  }
}

void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length) {
  if (matrix->imValues) {
    memcpy(&matrix->imValues[row * matrix->columns], vector,
           length * sizeof(floatt));
  }
}

void SetReVector(math::Matrix* matrix, uintt column, floatt* vector) {
  SetReVector(matrix, column, vector, matrix->rows);
}

void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector) {
  SetTransposeReVector(matrix, row, vector, matrix->columns);
}

void SetImVector(math::Matrix* matrix, uintt column, floatt* vector) {
  SetImVector(matrix, column, vector, matrix->rows);
}

void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector) {
  SetTransposeImVector(matrix, row, vector, matrix->columns);
}

void GetMatrixStr(std::string& text, const math::Matrix* matrix) {
  matrixUtils::PrintMatrix(text, matrix);
}

void GetReMatrixStr(std::string& text, const math::Matrix* matrix) {
  matrixUtils::PrintMatrix(text, matrix);
}

void GetImMatrixStr(std::string& str, const math::Matrix* matrix) {
  str = "";
  if (matrix == NULL) {
    return;
  }
  std::stringstream sstream;
  str += "[";
  for (int fb = 0; fb < matrix->rows; fb++) {
    for (int fa = 0; fa < matrix->columns; fa++) {
      sstream << matrix->imValues[fb * matrix->columns + fa];
      str += sstream.str();
      sstream.str("");
      if (fa != matrix->columns - 1) {
        str += ",";
      }
      if (fa == matrix->columns - 1 && fb != matrix->rows - 1) {
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
  if (revector != NULL) {
    GetReVector(revector, length, matrix, column);
  }

  if (imvector != NULL) {
    GetImVector(imvector, length, matrix, column);
  }
}

void GetTransposeVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  if (vector->reValues != NULL) {
    GetTransposeReVector(vector, matrix, column);
  }
  if (vector->imValues != NULL) {
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
  if (matrix->reValues) {
    for (uintt fa = 0; fa < length; fa++) {
      vector[fa] = matrix->reValues[column + matrix->columns * fa];
    }
  }
}

void GetTransposeReVector(floatt* vector, uint length, math::Matrix* matrix, uint row)
{
  if (matrix->reValues) {
    memcpy(vector, &matrix->reValues[row * matrix->columns],
           length * sizeof(floatt));
  }
}

void GetImVector(floatt* vector, uint length, math::Matrix* matrix, uint column)
{
  if (matrix->imValues) {
    for (uintt fa = 0; fa < length; fa++) {
      vector[fa] = matrix->imValues[column + matrix->columns * fa];
    }
  }
}

void GetTransposeImVector(floatt* vector, uint length, math::Matrix* matrix, uint row)
{
  if (matrix->imValues) {
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

floatt SmallestDiff(math::Matrix* matrix, math::Matrix* matrix1) {
  floatt diff = matrix->reValues[0] - matrix1->reValues[0];
  for (uintt fa = 0; fa < matrix->columns; fa++) {
    for (uintt fb = 0; fb < matrix->rows; fb++) {
      uintt index = fa + fb * matrix->columns;
      floatt diff1 = matrix->reValues[index] - matrix1->reValues[index];
      if (diff1 < 0) {
        diff1 = -diff1;
      }
      if (diff > diff1) {
        diff = diff1;
      }
    }
  }
  return diff;
}

floatt LargestDiff(math::Matrix* matrix, math::Matrix* matrix1) {
  floatt diff = matrix->reValues[0] - matrix1->reValues[0];
  for (uintt fa = 0; fa < matrix->columns; fa++) {
    for (uintt fb = 0; fb < matrix->rows; fb++) {
      uintt index = fa + fb * matrix->columns;
      floatt diff1 = matrix->reValues[index] - matrix1->reValues[index];
      if (diff1 < 0) {
        diff1 = -diff1;
      }
      if (diff < diff1) {
        diff = diff1;
      }
    }
  }
  return diff;
}

void SetIdentity(math::Matrix* matrix) {
  host::SetDiagonalReMatrix(matrix, 1);
  host::SetImZero(matrix);
}

void SetReZero(math::Matrix* matrix) {
  if (matrix->reValues) {
    memset(matrix->reValues, 0,
           matrix->columns * matrix->rows * sizeof(floatt));
  }
}

void SetImZero(math::Matrix* matrix) {
  if (matrix->imValues) {
    memset(matrix->imValues, 0,
           matrix->columns * matrix->rows * sizeof(floatt));
  }
}

void SetZero(math::Matrix* matrix) {
  SetReZero(matrix);
  SetImZero(matrix);
}

void SetIdentityMatrix(math::Matrix* matrix) {
  SetDiagonalReMatrix(matrix, 1);
  SetImZero(matrix);
}

bool IsEquals(math::Matrix* transferMatrix2, math::Matrix* transferMatrix1,
              floatt diff) {
  for (uintt fa = 0; fa < transferMatrix2->columns; fa++) {
    for (uintt fb = 0; fb < transferMatrix2->rows; fb++) {
      floatt p = transferMatrix2->reValues[fa + transferMatrix2->columns * fb] -
                 transferMatrix1->reValues[fa + transferMatrix1->columns * fb];
      if (p < -diff || p > diff) {
        return false;
      }
    }
  }
  return true;
}

floatt GetTrace(math::Matrix* matrix) {
  floatt o = 1.;
  for (uintt fa = 0; fa < matrix->columns; ++fa) {
    floatt v = matrix->reValues[fa * matrix->columns + fa];
    if (-MATH_VALUE_LIMIT < v && v < MATH_VALUE_LIMIT) {
      v = 0;
    }
    o = o * v;
  }
  return o;
}

void SetDiagonalMatrix(math::Matrix* matrix, floatt a) {
  SetDiagonalReMatrix(matrix, a);
  SetDiagonalImMatrix(matrix, a);
}

void SetDiagonalReMatrix(math::Matrix* matrix, floatt a) {
  if (matrix->reValues) {
    fillRePart(matrix, 0);
    for (int fa = 0; fa < matrix->columns; fa++) {
      matrix->reValues[fa * matrix->columns + fa] = a;
    }
  }
}

void SetDiagonalImMatrix(math::Matrix* matrix, floatt a) {
  if (matrix->imValues) {
    fillImPart(matrix, 0);
    for (int fa = 0; fa < matrix->columns; fa++) {
      matrix->imValues[fa * matrix->columns + fa] = a;
    }
  }
}

math::MatrixInfo GetMatrixInfo(const math::Matrix* matrix) {
  return math::MatrixInfo(matrix->reValues != NULL, matrix->imValues != NULL,
                          matrix->columns, matrix->rows);
}

void SetSubs(math::Matrix* matrix, uintt subcolumns, uintt subrows) {
  SetSubColumns(matrix, subcolumns);
  SetSubRows(matrix, subrows);
}

void SetSubColumns(math::Matrix* matrix, uintt subcolumns) {
  if (subcolumns == MATH_UNDEFINED) {
    matrix->columns = matrix->realColumns;
  } else {
    matrix->columns = subcolumns;
    debugAssert(matrix->columns <= matrix->realColumns);
  }
}

void SetSubRows(math::Matrix* matrix, uintt subrows) {
  if (subrows == MATH_UNDEFINED) {
    matrix->rows = matrix->realRows;
  } else {
    matrix->rows = subrows;
    debugAssert(matrix->rows <= matrix->realRows);
  }
}

void SetSubsSafe(math::Matrix* matrix, uintt subcolumns, uintt subrows) {
  SetSubColumnsSafe(matrix, subcolumns);
  SetSubRowsSafe(matrix, subrows);
}

void SetSubColumnsSafe(math::Matrix* matrix, uintt subcolumns) {
  if (subcolumns == MATH_UNDEFINED || matrix->columns < subcolumns) {
    matrix->columns = matrix->realColumns;
  } else {
    matrix->columns = subcolumns;
  }
}

void SetSubRowsSafe(math::Matrix* matrix, uintt subrows) {
  if (subrows == MATH_UNDEFINED || matrix->rows < subrows) {
    matrix->rows = matrix->realRows;
  } else {
    debugAssert(matrix->rows <= matrix->realRows);
  }
}

struct FileHeader {
  math::MatrixInfo matrixInfo;
  uint32_t sizeofbool;
  uint32_t sizeofuintt;
  uint32_t sizeoffloatt;
};

FileHeader loadHeader(FILE* file) {

  FileHeader fh;

  auto readValue = [file](uint32_t& value) {
    fread(&value, sizeof(value), 1, file);
  };

  auto readBuffer = [file](void* buffer, uint32_t size) {
    fread(buffer, size, 1, file);
  };

  readValue(fh.sizeofbool);
  readValue(fh.sizeofuintt);
  readValue(fh.sizeoffloatt);

  if (fh.sizeofbool == sizeof(bool) && fh.sizeofuintt == sizeof(uintt)) {
    readBuffer(&fh.matrixInfo, sizeof(fh.matrixInfo));
  } else {
    readBuffer(&fh.matrixInfo.m_matrixDim.columns, fh.sizeofuintt);
    readBuffer(&fh.matrixInfo.m_matrixDim.rows, fh.sizeofuintt);
    readBuffer(&fh.matrixInfo.isRe, fh.sizeofbool);
    readBuffer(&fh.matrixInfo.isIm, fh.sizeofbool);
  }

  return fh;
}

math::Matrix* ReadMatrix(const std::string& path, const MatrixEx& a_matrixEx) {

  FILE* file = fopen(path.c_str(), "rb");

  if (file == NULL) {
    return NULL;
  }

  MatrixEx matrixEx = a_matrixEx;

  FileHeader fileHeader = loadHeader(file);

  const math::MatrixInfo lMatrixInfo = fileHeader.matrixInfo;
  const uint32_t sizeoffloatt = fileHeader.sizeoffloatt;

  const uintt columns = lMatrixInfo.m_matrixDim.columns;
  const uintt rows = lMatrixInfo.m_matrixDim.rows;

  if (!adjustRows(matrixEx, rows) || !adjustColumns(matrixEx, columns)) {
    fclose(file);
    return nullptr;
  }

  math::MatrixInfo matrixInfo = lMatrixInfo;

  bool isIdentical = matrixEx.beginRow == 0 && matrixEx.beginColumn == 0 &&
          erow(matrixEx) == rows && ecolumn(matrixEx) == columns;

  matrixInfo.m_matrixDim.columns = matrixEx.columnsLength;
  matrixInfo.m_matrixDim.rows = matrixEx.rowsLength;

  math::Matrix* matrix = host::NewMatrix(matrixInfo);

  size_t lcounts = lMatrixInfo.m_matrixDim.columns * lMatrixInfo.m_matrixDim.rows;
  size_t lsize = sizeoffloatt * lcounts;

  std::shared_ptr<floatt> sectionPtr(nullptr);
  if (isIdentical == false) {
    sectionPtr.reset(new floatt[lcounts], std::default_delete<floatt[]>());
  }

  auto loadSection = [isIdentical, sizeoffloatt, lMatrixInfo,
       lcounts, lsize, matrixEx, sectionPtr, file] (floatt* section)
  {
    if (section == nullptr) { return; }

    floatt* sectionTmp = section;

    if (isIdentical == false) {
      sectionTmp = sectionPtr.get();
    }

    if (sizeoffloatt == sizeof(floatt)) {
      fread(sectionTmp, lsize, 1, file);
    } else {
      std::unique_ptr<char[]> buffer(new char[lsize]);
      fread(buffer.get(), lsize, 1, file);
      for (size_t idx = 0; idx < lcounts; ++idx) {
        memcpy(&sectionTmp[idx], &buffer[idx * sizeoffloatt], sizeoffloatt);
      }
    }

    if (isIdentical == false) {
      for (uint idx = 0; idx < matrixEx.rowsLength; ++idx) {
        memcpy(&section[idx * matrixEx.columnsLength],
               &sectionTmp[(idx + matrixEx.beginRow) * lMatrixInfo.m_matrixDim.columns + matrixEx.beginColumn],
               matrixEx.columnsLength * sizeof(floatt));
      }
    }
  };

  loadSection(matrix->reValues);
  loadSection(matrix->imValues);

  fclose(file);

  return matrix;
}

math::Matrix* ReadMatrix(const std::string& path) {
  MatrixEx me;

  me.beginColumn = 0;
  me.columnsLength = static_cast<uintt>(-1);

  me.beginRow = 0;
  me.rowsLength = static_cast<uintt>(-1);

  return ReadMatrix(path, me);
}

math::Matrix* ReadRowVector(const std::string& path, size_t index) {
  MatrixEx me;

  me.beginColumn = 0;
  me.columnsLength = static_cast<uintt>(-1);

  me.beginRow = index;
  me.rowsLength = 1;

  return ReadMatrix(path, me);
}

math::Matrix* ReadColumnVector(const std::string& path, size_t index) {
  MatrixEx me;

  me.beginColumn = index;
  me.columnsLength = 1;

  me.beginRow = 0;
  me.rowsLength = static_cast<uintt>(-1);

  return ReadMatrix(path, me);
}

bool WriteMatrix(const std::string& path, const math::Matrix* matrix) {

  FILE* file = fopen(path.c_str(), "wb");

  if (file == NULL) {
    return false;
  }

  const size_t size = sizeof(floatt) * matrix->columns * matrix->rows;

  uint32_t sizeofbool = sizeof(bool);
  uint32_t sizeofuintt = sizeof(uintt);
  uint32_t sizeoffloatt = sizeof(floatt);

  fwrite(&sizeofbool, sizeof(uint32_t), 1, file);
  fwrite(&sizeofuintt, sizeof(uint32_t), 1, file);
  fwrite(&sizeoffloatt, sizeof(uint32_t), 1, file);

  math::MatrixInfo matrixInfo = host::GetMatrixInfo(matrix);

  fwrite(&matrixInfo, sizeof(matrixInfo), 1, file);

  if (matrixInfo.isRe) {
    fwrite(matrix->reValues, size, 1, file);
  }

  if (matrixInfo.isIm) {
    fwrite(matrix->imValues, size, 1, file);
  }

  fclose(file);

  return true;
}
};
