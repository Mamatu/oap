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

#include "HostMatrixModules.h"
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
                                       const math::Matrix* matrix) {}

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

void HostMatrixUtils::setDiagonalReMatrix(math::Matrix* matrix, floatt value) {
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
