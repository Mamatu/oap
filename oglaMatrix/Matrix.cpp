/*
 * File:   Matrix.cpp
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#include "Matrix.h"
#include <stdlib.h>
#include <map>

#ifdef DEBUG

enum Type { Re, Im };

typedef std::pair<uintt, uintt> MatrixDim;
typedef std::pair<floatt, Type> MatrixValue;
typedef std::pair<MatrixDim, MatrixValue> MatrixElement;
typedef std::multimap<const math::Matrix*, MatrixElement> MatrixElements;

MatrixElements globalMatrixElementsSet;
MatrixElements globalMatrixElementsGet;

namespace test {

bool isElement(const MatrixElements& matrixElements, const math::Matrix* matrix,
               uintt column, uintt row, Type type) {
  std::pair<MatrixElements::const_iterator, MatrixElements::const_iterator>
      range = matrixElements.equal_range(matrix);
  for (MatrixElements::const_iterator it = range.first; it != range.second;
       ++it) {
    const uintt lcolumn = (*it).second.first.first;
    const uintt lrow = (*it).second.first.second;
    if (lcolumn == column && lrow == row && it->second.second.second == type) {
      return true;
    }
  }
  return false;
}

void addElement(MatrixElements& matrixElements, const math::Matrix* matrix,
                uintt column, uintt row, floatt value, Type type) {
  matrixElements.insert(std::pair<const math::Matrix*, MatrixElement>(
      matrix, MatrixElement(MatrixDim(column, row), MatrixValue(value, type))));
}

void removeElement(MatrixElements& matrixElements, const math::Matrix* matrix) {
  MatrixElements::iterator it = matrixElements.find(matrix);
  if (it != matrixElements.end()) {
    matrixElements.erase(it);
  }
}

void reset(const math::Matrix* matrix) {
  removeElement(globalMatrixElementsSet, matrix);
  removeElement(globalMatrixElementsGet, matrix);
}

void setRe(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsSet, matrix, column, row, value, Re);
}

void setIm(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsSet, matrix, column, row, value, Im);
}

void getRe(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsGet, matrix, column, row, value, Re);
}

void getIm(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsGet, matrix, column, row, value, Im);
}

bool wasSetRe(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsSet, matrix, column, row, Re);
}

bool wasSetIm(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsSet, matrix, column, row, Im);
}

bool wasGetRe(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsGet, matrix, column, row, Re);
}
bool wasGetIm(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsGet, matrix, column, row, Im);
}

bool areAllElements(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                    uintt brow, uintt erow,
                    bool (*funcptr)(const math::Matrix* matrix, uintt column,
                                    uintt row)) {
  for (uintt column = bcolumn; column < ecolumn; ++column) {
    for (uintt row = brow; row < erow; ++row) {
      if ((*funcptr)(matrix, column, row) == false) {
        return false;
      }
    }
  }
  return true;
}

bool wasSetRangeRe(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow) {
  return areAllElements(matrix, bcolumn, ecolumn, brow, erow, wasSetRe);
}
bool wasSetRangeIm(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow) {
  return areAllElements(matrix, bcolumn, ecolumn, brow, erow, wasSetIm);
}

bool wasSetAllRe(const math::Matrix* matrix) {
  return wasSetRangeRe(matrix, 0, matrix->columns, 0, matrix->rows);
}

bool wasSetAllIm(const math::Matrix* matrix) {
  return wasSetRangeIm(matrix, 0, matrix->columns, 0, matrix->rows);
}

bool wasGetRangeRe(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow) {
  return areAllElements(matrix, bcolumn, ecolumn, brow, erow, wasGetRe);
}

bool wasGetRangeIm(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow) {
  return areAllElements(matrix, bcolumn, ecolumn, brow, erow, wasGetIm);
}

bool wasGetAllRe(const math::Matrix* matrix) {
  return wasGetRangeRe(matrix, 0, matrix->columns, 0, matrix->rows);
}

bool wasGetAllIm(const math::Matrix* matrix) {
  return wasGetRangeIm(matrix, 0, matrix->columns, 0, matrix->rows);
}
};

#endif
