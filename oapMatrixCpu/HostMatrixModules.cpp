/*
 * Copyright 2016 Marcin Matula
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




#include "HostMatrixModules.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <linux/fs.h>
#include "MatrixUtils.h"
#include "ArrayTools.h"
#include "Writer.h"
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

#elif RELEASE

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillRePart(math::Matrix* output, floatt value) {
  math::Memset(output->reValues, value, output->columns * output->rows);
}

inline void fillImPart(math::Matrix* output, floatt value) {
  math::Memset(output->imValues, value, output->columns * output->rows);
}

inline void fillMatrix(math::Matrix* output, floatt value) {
  if (output->reValues) {
    fillRePart(output, value);
  }
  if (output->imValues) {
    fillImPart(output, value);
  }
}

HostMatrixAllocator::HostMatrices HostMatrixAllocator::hostMatrices;
utils::sync::Mutex HostMatrixAllocator::mutex;

void _memset(floatt* s, floatt value, int n) {
  for (int fa = 0; fa < n; fa++) {
    memcpy(s + fa, &value, sizeof(floatt));
  }
}

math::Matrix* HostMatrixAllocator::createHostMatrix(math::Matrix* matrix,
                                                    uintt columns, uintt rows,
                                                    floatt* values,
                                                    floatt** valuesPtr) {
  *valuesPtr = values;
  matrix->realColumns = columns;
  matrix->columns = columns;
  matrix->realRows = rows;
  matrix->rows = rows;
  HostMatrixAllocator::mutex.lock();
  HostMatrixAllocator::hostMatrices.push_back(matrix);
  HostMatrixAllocator::mutex.unlock();
  return matrix;
}

HostMatrixCopier::HostMatrixCopier()
    : MatrixCopier(HostMatrixModules::GetInstance()) {}

HostMatrixCopier::~HostMatrixCopier() {}

inline void HostMatrixAllocator::initMatrix(math::Matrix* matrix) {
  matrix->columns = 0;
  matrix->rows = 0;
  matrix->imValues = NULL;
  matrix->reValues = NULL;
}

math::Matrix* HostMatrixAllocator::createHostReMatrix(uintt columns, uintt rows,
                                                      floatt* values) {
  math::Matrix* matrix = new math::Matrix();
  initMatrix(matrix);
  matrix->realColumns = columns;
  matrix->columns = columns;
  matrix->realRows = rows;
  matrix->rows = rows;
  return createHostMatrix(matrix, columns, rows, values, &matrix->reValues);
}

math::Matrix* HostMatrixAllocator::createHostImMatrix(uintt columns, uintt rows,
                                                      floatt* values) {
  math::Matrix* matrix = new math::Matrix();
  initMatrix(matrix);
  return createHostMatrix(matrix, columns, rows, values, &matrix->imValues);
}

math::Matrix* HostMatrixAllocator::newMatrix(uintt columns, uintt rows,
                                             floatt value) {
  return host::NewMatrix(columns, rows, value);
}

math::Matrix* HostMatrixAllocator::newReMatrix(uintt columns, uintt rows,
                                               floatt value) {
  return host::NewReMatrix(columns, rows, value);
}

math::Matrix* HostMatrixAllocator::newImMatrix(uintt columns, uintt rows,
                                               floatt value) {
  return host::NewImMatrix(columns, rows, value);
}

void HostMatrixAllocator::deleteMatrix(math::Matrix* matrix) {
  host::DeleteMatrix(matrix);
}

void HostMatrixCopier::copyMatrixToMatrix(math::Matrix* dst,
                                          const math::Matrix* src) {
  host::CopyMatrix(dst, src);
}

void HostMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst,
                                              const math::Matrix* src) {
  host::CopyRe(dst, src);
}

void HostMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst,
                                              const math::Matrix* src) {
  host::CopyIm(dst, src);
}

void HostMatrixCopier::copy(floatt* dst, const floatt* src, uintt length) {
  memcpy(dst, src, length * sizeof(floatt));
}

HostMatrixUtils::HostMatrixUtils()
    : MatrixUtils(HostMatrixModules::GetInstance()) {}

HostMatrixUtils::~HostMatrixUtils() {}

HostMatrixAllocator::HostMatrixAllocator()
    : MatrixAllocator(HostMatrixModules::GetInstance()) {}

HostMatrixAllocator::~HostMatrixAllocator() {}

HostMatrixPrinter::HostMatrixPrinter()
    : MatrixPrinter(HostMatrixModules::GetInstance()) {}

HostMatrixPrinter::~HostMatrixPrinter() {}

bool HostMatrixAllocator::isMatrix(math::Matrix* matrix) {
  HostMatrices::iterator it =
      std::find(hostMatrices.begin(), hostMatrices.end(), matrix);
  return (it != hostMatrices.end());
}
#ifdef MATLAB

void HostMatrixPrinter::getMatrixStr(std::string& str,
                                     const math::Matrix* matrix) {
  str = "";
  if (matrix == NULL) {
    return;
  }
  std::stringstream sstream;
  str += "{";
  for (int fb = 0; fb < matrix->rows; fb++) {
    for (int fa = 0; fa < matrix->columns; fa++) {
      if (fa == 0) {
        str += "{";
      }
      sstream << matrix->reValues[fb * matrix->columns + fa];
      str += sstream.str();
      sstream.str("");
      sstream << matrix->imValues[fb * matrix->columns + fa];
      str += "+" + sstream.str() + "i";
      sstream.str("");
      if (fa != matrix->columns - 1) {
        str += ",";
      }
      if (fa == matrix->columns - 1 /*&& fb != matrix->rows - 1*/) {
        str += "}";
      }
    }
  }
  str += "}";
}
#endif

void HostMatrixPrinter::getMatrixStr(std::string& str,
                                     const math::Matrix* matrix) {
  str = "";
  if (matrix == NULL) {
    return;
  }
  std::stringstream sstream;
  str += "[";
  for (int fb = 0; fb < matrix->rows; fb++) {
    for (int fa = 0; fa < matrix->columns; fa++) {
      sstream << matrix->reValues[fb * matrix->columns + fa];
      str += "(" + sstream.str();
      sstream.str("");
      sstream << matrix->imValues[fb * matrix->columns + fa];
      str += "," + sstream.str() + "i)";
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

void HostMatrixPrinter::getReMatrixStr(std::string& text,
                                       const math::Matrix* matrix) {
  text = "";
  if (matrix == NULL) {
    return;
  }
  std::stringstream sstream;
  text += "\n[";
  char buffer[128];
  memset(buffer, 0, 128 * sizeof(char));
  for (int fb = 0; fb < matrix->rows; fb++) {
    for (int fa = 0; fa < matrix->columns; fa++) {
      sprintf(buffer, "%lf", matrix->reValues[fb * matrix->columns + fa]);
      text += buffer;
      memset(buffer, 0, 128 * sizeof(char));
      if (fa != matrix->columns - 1) {
        text += ",";
      }
      if (fa == matrix->columns - 1 && fb != matrix->rows - 1) {
        text += "\n";
      }
    }
  }
  text += "]\n";
}

void HostMatrixPrinter::getImMatrixStr(std::string& str,
                                       const math::Matrix* matrix) {

}

void HostMatrixUtils::getReValues(floatt* dst, math::Matrix* matrix,
                                  uintt index, uintt length) {
  if (matrix->reValues) {
    memcpy(dst, &matrix->reValues[index], length);
  }
}

void HostMatrixUtils::getImValues(floatt* dst, math::Matrix* matrix,
                                  uintt index, uintt length) {
  if (matrix->imValues) {
    memcpy(dst, &matrix->imValues[index], length);
  }
}

void HostMatrixUtils::setReValues(math::Matrix* matrix, floatt* src,
                                  uintt index, uintt length) {
  if (matrix->reValues) {
    memcpy(&matrix->reValues[index], src, length);
  }
}

void HostMatrixUtils::setImValues(math::Matrix* matrix, floatt* src,
                                  uintt index, uintt length) {
  if (matrix->imValues) {
    memcpy(&matrix->imValues[index], src, length);
  }
}

void HostMatrixUtils::setZeroReMatrix(math::Matrix* matrix) {
  if (matrix->reValues) {
    fillRePart(matrix, 0);
  }
}

void HostMatrixUtils::setZeroImMatrix(math::Matrix* matrix) {
  if (matrix->imValues) {
    fillImPart(matrix, 0);
  }
}

void HostMatrixUtils::setDiagonalReMatrix(math::Matrix* matrix, floatt value) {
  host::SetDiagonalReMatrix(matrix, value);
}

void HostMatrixUtils::setDiagonalImMatrix(math::Matrix* matrix, floatt value) {
  if (matrix->imValues) {
    fillImPart(matrix, 0);
    for (int fa = 0; fa < matrix->columns; fa++) {
      matrix->imValues[fa * matrix->columns + fa] = value;
    }
  }
}

void HostMatrixCopier::setTransposeReVector(math::Matrix* matrix, uintt row,
                                            floatt* vector, uintt length) {
  host::SetTransposeReVector(matrix, row, vector, length);
}

void HostMatrixCopier::setReVector(math::Matrix* matrix, uintt column,
                                   floatt* vector, uintt length) {
  host::SetReVector(matrix, column, vector, length);
}

void HostMatrixCopier::setTransposeImVector(math::Matrix* matrix, uintt row,
                                            floatt* vector, uintt length) {
  host::SetTransposeImVector(matrix, row, vector, length);
}

void HostMatrixCopier::setImVector(math::Matrix* matrix, uintt column,
                                   floatt* vector, uintt length) {
  host::SetImVector(matrix, column, vector, length);
}

void HostMatrixCopier::getTransposeReVector(floatt* vector, uintt length,
                                            math::Matrix* matrix, uintt row) {
  host::GetTransposeReVector(vector, length, matrix, row);
}

void HostMatrixCopier::getReVector(floatt* vector, uintt length,
                                   math::Matrix* matrix, uintt column) {
  host::GetReVector(vector, length, matrix, column);
}

void HostMatrixCopier::getTransposeImVector(floatt* vector, uintt length,
                                            math::Matrix* matrix, uintt row) {
  host::GetTransposeImVector(vector, length, matrix, row);
}

void HostMatrixCopier::setVector(math::Matrix* matrix, uintt column,
                                 math::Matrix* vector, uintt rows) {
  setReVector(matrix, column, vector->reValues, rows);
  setImVector(matrix, column, vector->imValues, rows);
}

void HostMatrixCopier::getVector(math::Matrix* vector, uintt rows,
                                 math::Matrix* matrix, uintt column) {
  host::GetReVector(vector->reValues, rows, matrix, column);
  host::GetImVector(vector->imValues, rows, matrix, column);
}

void HostMatrixCopier::getImVector(floatt* vector, uintt length,
                                   math::Matrix* matrix, uintt column) {
  host::GetImVector(vector, length, matrix, column);
}

math::Matrix* HostMatrixAllocator::newMatrixFromAsciiFile(const char* path) {
  FILE* file = fopen(path, "r");
  math::Matrix* matrix = NULL;
  int stackCounter = 0;
  if (file) {
    bool is = false;
    floatt* values = NULL;
    int valuesSize = 0;
    uintt columns = 0;
    uintt rows = 0;
    std::string buffer = "";
    fseek(file, 0, SEEK_END);
    long int size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* text = new char[size];
    fread(text, size * sizeof(char), 1, file);
    for (int fa = 0; fa < size; fa++) {
      char sign = text[fa];
      if (sign == '[') {
        stackCounter++;
      } else if (sign == ']') {
        columns++;
        is = true;
        stackCounter--;
      } else if (sign == ',') {
        if (is == false) {
          rows++;
        }
        floatt value = atof(buffer.c_str());
        buffer.clear();
        ArrayTools::add(&values, valuesSize, value);
      } else {
        buffer += sign;
      }
      if (stackCounter == 0) {
        break;
      }
    }
    columns--;
    rows++;
    matrix = createHostReMatrix(columns, rows, values);
    delete[] text;
  }
  return matrix;
}

math::Matrix* HostMatrixAllocator::newMatrixFromBinaryFile(const char* path) {
  FILE* file = fopen(path, "rb");
  math::Matrix* matrix = NULL;
  if (file) {
    int size = 0;
    uintt columns = 0;
    uintt rows = 0;
    fread(&size, sizeof(int), 1, file);
    fread(&columns, sizeof(int), 1, file);
    fread(&rows, sizeof(int), 1, file);
    matrix = this->newReMatrix(columns, rows);
    fread(matrix->reValues, sizeof(floatt) * columns * rows, 1, file);
    fclose(file);
  }
  return matrix;
}

uintt HostMatrixUtils::getColumns(const math::Matrix* matrix) const {
  return matrix->columns;
}

uintt HostMatrixUtils::getRows(const math::Matrix* matrix) const {
  return matrix->rows;
}

bool HostMatrixUtils::isMatrix(const math::Matrix* matrix) const {
  return matrix != NULL;
}

bool HostMatrixUtils::isReMatrix(const math::Matrix* matrix) const {
  return matrix != NULL && matrix->reValues != NULL;
}

bool HostMatrixUtils::isImMatrix(const math::Matrix* matrix) const {
  return matrix != NULL && matrix->imValues != NULL;
}

HostMatrixModules::HostMatrixModules() {
  m_hma = NULL;
  m_hmc = NULL;
  m_hmp = NULL;
  m_hmu = NULL;
}

HostMatrixModules::~HostMatrixModules() {}

HostMatrixAllocator* HostMatrixModules::getMatrixAllocator() { return m_hma; }

HostMatrixCopier* HostMatrixModules::getMatrixCopier() { return m_hmc; }

HostMatrixUtils* HostMatrixModules::getMatrixUtils() { return m_hmu; }

HostMatrixPrinter* HostMatrixModules::getMatrixPrinter() { return m_hmp; }

HostMatrixModules* HostMatrixModules::hostMatrixModule = NULL;

HostMatrixModules* HostMatrixModules::GetInstance() {
  if (NULL == HostMatrixModules::hostMatrixModule) {
    HostMatrixModules::hostMatrixModule = new HostMatrixModules();
    HostMatrixModules::hostMatrixModule->m_hma = new HostMatrixAllocator();
    HostMatrixModules::hostMatrixModule->m_hmc = new HostMatrixCopier();
    HostMatrixModules::hostMatrixModule->m_hmp = new HostMatrixPrinter();
    HostMatrixModules::hostMatrixModule->m_hmu = new HostMatrixUtils();
  }
  return HostMatrixModules::hostMatrixModule;
}

namespace host {

math::Matrix* NewMatrixCopy(const math::Matrix* matrix) {
  math::Matrix* output = NULL;
  if (matrix->reValues && matrix->imValues) {
    output = NewMatrix(matrix->columns, matrix->rows);
    CopyMatrix(output, matrix);
  } else if (matrix->reValues) {
    output = NewReMatrix(matrix->columns, matrix->rows);
    CopyRe(output, matrix);
  } else if (matrix->imValues) {
    output =
        HostMatrixModules::GetInstance()->getMatrixAllocator()->newImMatrix(
            matrix->columns, matrix->rows);
    CopyIm(output, matrix);
  }
  return output;
}

math::Matrix* NewMatrix(const math::Matrix* matrix, floatt value) {
  math::Matrix* output = NULL;
  if (matrix->reValues != NULL && matrix->imValues != NULL) {
    output = NewMatrix(matrix->columns, matrix->rows, value);
  } else if (matrix->reValues != NULL) {
    output = NewReMatrix(matrix->columns, matrix->rows, value);
  } else if (matrix->imValues != NULL) {
    output = NewImMatrix(matrix->columns, matrix->rows, value);
  } else {
    return NULL;
  }
}

math::Matrix* NewMatrix(const math::Matrix* matrix, uintt columns, uintt rows,
                        floatt value) {
  math::Matrix* output = NULL;
  if (matrix->reValues != NULL && matrix->imValues != NULL) {
    output = NewMatrix(columns, rows, value);
  } else if (matrix->reValues != NULL) {
    output = NewReMatrix(columns, rows, value);
  } else if (matrix->imValues != NULL) {
    output = NewImMatrix(columns, rows, value);
  } else {
    return NULL;
  }
}

math::Matrix* NewMatrixCopy(uintt columns, uintt rows, floatt* reArray,
                            floatt* imArray) {
  debugAssert(reArray != NULL || imArray != NULL);
  math::Matrix* output =
      HostMatrixModules::GetInstance()->getMatrixAllocator()->newMatrix(columns,
                                                                        rows);
  if (NULL != reArray) {
    HostMatrixModules::GetInstance()->getMatrixCopier()->copy(
        output->reValues, reArray, columns * rows);
  }
  if (NULL != imArray) {
    HostMatrixModules::GetInstance()->getMatrixCopier()->copy(
        output->imValues, imArray, columns * rows);
  }
  return output;
}

math::Matrix* NewReMatrixCopy(uintt columns, uintt rows, floatt* array) {
  math::Matrix* output =
      HostMatrixModules::GetInstance()->getMatrixAllocator()->newReMatrix(
          columns, rows);
  HostMatrixModules::GetInstance()->getMatrixCopier()->copy(
      output->reValues, array, columns * rows);
  return output;
}

math::Matrix* NewImMatrixCopy(uintt columns, uintt rows, floatt* array) {
  math::Matrix* output =
      HostMatrixModules::GetInstance()->getMatrixAllocator()->newImMatrix(
          columns, rows);
  HostMatrixModules::GetInstance()->getMatrixCopier()->copy(
      output->reValues, array, columns * rows);
  return output;
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
  output->reValues = new floatt[length];
  output->imValues = NULL;
  output->realColumns = columns;
  output->columns = columns;
  output->realRows = rows;
  output->rows = rows;
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

void PrintMatrix(const std::string& text, const math::Matrix* matrix) {
  std::string output = text + " ";
  matrixUtils::PrintMatrix(output, matrix);
  printf("%s HOST \n", output.c_str());
}

void PrintMatrix(const math::Matrix* matrix) { PrintMatrix("", matrix); }

void Copy(math::Matrix* dst, const math::Matrix* src,
          const SubMatrix& subMatrix, uintt column, uintt row) {
  HostMatrixCopier* copier =
      HostMatrixModules::GetInstance()->getMatrixCopier();
  uintt rows = dst->rows;
  uintt columns2 = subMatrix.m_columns;
  for (uintt fa = 0; fa < rows; fa++) {
    uintt fa1 = fa + subMatrix.m_brow;
    if (fa < row) {
      copier->copy(dst->reValues + fa * dst->columns,
                   src->reValues + (fa1)*columns2, column);
      copier->copy(dst->reValues + column + fa * dst->columns,
                   src->reValues + (1 + column) + fa * columns2,
                   (columns2 - column));
    } else if (fa >= row) {
      copier->copy(dst->reValues + fa * dst->columns,
                   &src->reValues[(fa1 + 1) * columns2], column);

      copier->copy(dst->reValues + column + fa * dst->columns,
                   &src->reValues[(fa1 + 1) * columns2 + column + 1],
                   (columns2 - column));
    }
  }
}

void Copy(math::Matrix* dst, const math::Matrix* src, uintt column, uintt row) {
  HostMatrixCopier* copier =
      HostMatrixModules::GetInstance()->getMatrixCopier();
  uintt rows = src->rows;
  uintt columns = src->columns;
  for (uintt fa = 0; fa < rows; fa++) {
    if (fa < row) {
      copier->copy(&dst->reValues[fa * dst->columns],
                   &src->reValues[fa * columns], column);
      if (column < src->columns - 1) {
        copier->copy(&dst->reValues[column + fa * dst->columns],
                     &src->reValues[(1 + column) + fa * columns],
                     (src->columns - (column + 1)));
      }
    } else if (fa > row) {
      copier->copy(&dst->reValues[(fa - 1) * dst->columns],
                   &src->reValues[fa * columns], column);
      if (column < src->columns - 1) {
        copier->copy(&dst->reValues[column + (fa - 1) * dst->columns],
                     &src->reValues[fa * columns + (column + 1)],
                     (src->columns - (column + 1)));
      }
    }
  }
}

void CopyMatrix(math::Matrix* dst, const math::Matrix* src) {
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  uintt bytesLength = length * sizeof(floatt);
  if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
    memcpy(dst->reValues, src->reValues, bytesLength);
  }
  if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
    memcpy(dst->imValues, src->imValues, bytesLength);
  }
}

void CopyRe(math::Matrix* dst, const math::Matrix* src) {
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
    memcpy(dst->reValues, src->reValues, length * sizeof(floatt));
  } else {
  }
}

void CopyIm(math::Matrix* dst, const math::Matrix* src) {
  const uintt length1 = dst->columns * dst->rows;
  const uintt length2 = src->columns * src->rows;
  const uintt length = length1 < length2 ? length1 : length2;
  if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
    memcpy(dst->imValues, src->imValues, length * sizeof(floatt));
  } else {
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
  HostMatrixModules::GetInstance()->getMatrixPrinter()->getReMatrixStr(text,
                                                                       matrix);
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

void GetReVector(floatt* vector, uintt length, math::Matrix* matrix,
                 uintt column) {
  if (matrix->reValues) {
    for (uintt fa = 0; fa < length; fa++) {
      vector[fa] = matrix->reValues[column + matrix->columns * fa];
    }
  }
}

void GetTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix,
                          uintt row) {
  if (matrix->reValues) {
    memcpy(vector, &matrix->reValues[row * matrix->columns],
           length * sizeof(floatt));
  }
}

void GetImVector(floatt* vector, uintt length, math::Matrix* matrix,
                 uintt column) {
  if (matrix->imValues) {
    for (uintt fa = 0; fa < length; fa++) {
      vector[fa] = matrix->imValues[column + matrix->columns * fa];
    }
  }
}

void GetTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix,
                          uintt row) {
  if (matrix->imValues) {
    memcpy(vector, &matrix->imValues[row * matrix->columns],
           length * sizeof(floatt));
  }
}

void GetReVector(floatt* vector, math::Matrix* matrix, uintt column) {
  GetReVector(vector, matrix->rows, matrix, column);
}

void GetTransposeReVector(floatt* vector, math::Matrix* matrix, uintt row) {
  GetTransposeReVector(vector, matrix->columns, matrix, row);
}

void GetImVector(floatt* vector, math::Matrix* matrix, uintt column) {
  GetImVector(vector, matrix->rows, matrix, column);
}

void GetTransposeImVector(floatt* vector, math::Matrix* matrix, uintt row) {
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
  HostMatrixModules::GetInstance()->getMatrixUtils()->setIdentityMatrix(matrix);
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

char* load(const char* path, uintt& _size) {
  FILE* file = fopen(path, "r");
  fseek(file, 0, SEEK_END);
  long int size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char* buffer = new char[size + 1];
  buffer[size] = 0;
  fread(buffer, size, 1, file);
  fclose(file);
  _size = size;
  return buffer;
}

void loadFloats(floatt* values, uintt count, char* data, unsigned int size,
                char separator, uintt skip) {
  char* ptr = &data[0];
  uintt index = 0;
  uintt index1 = 0;
  bool c = false;
  if (skip == index1) {
    c = true;
  }
  for (uint fa = 0; fa < size; ++fa) {
    if (data[fa] == separator) {
      char* ptr1 = &data[fa];
      if (c) {
        std::string s(ptr, ptr1 - ptr);
        floatt f = atof(s.c_str());
        values[index] = f;
      }
      ptr = &data[fa + 1];
      index++;
      if (index == count) {
        index = 0;
        index1++;
        if (skip == index1) {
          c = true;
        } else if (skip + 1 == index1) {
          return;
        }
      }
    }
  }
}

math::Matrix* LoadMatrix(uintt columns, uintt rows, const char* repath,
                         const char* impath) {
  math::Matrix* matrix = NewMatrix(columns, rows, 0);
  LoadMatrix(matrix, repath, impath);
  return matrix;
}

void LoadMatrix(math::Matrix* matrix, const char* repath, const char* impath) {
  LoadMatrix(matrix, repath, impath, 0);
}

void LoadMatrix(math::Matrix* matrix, const char* repath, const char* impath,
                uintt skipCount) {
  if (NULL != matrix) {
    uintt length = matrix->columns * matrix->rows;
    uintt size;
    char* b = NULL;
    if (NULL != repath) {
      b = load(repath, size);
      loadFloats(matrix->reValues, length, b, size, ',', skipCount);
      delete[] b;
    }
    if (NULL != impath) {
      b = load(impath, size);
      loadFloats(matrix->imValues, length, b, size, ',', skipCount);
      delete[] b;
    }
  }
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

  if ((iscolumns && isrows) == false) {
    size_t sq = sqrt(pairRe.second);
    columns = sq;
    rows = sq;
    iscolumns = true;
    isrows = true;
  } else if (iscolumns && !isrows) {
    rows = pairRe.second - columns;
    isrows = true;
  } else if (isrows && !iscolumns) {
    columns = pairRe.second - rows;
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
};
