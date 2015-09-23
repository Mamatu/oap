/*
 * File:   Matrix.cpp
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#include "Matrix.h"
#include <stdlib.h>
#include <map>
#include "ThreadUtils.h"

enum Type { Re, Im };

typedef std::pair<uintt, uintt> MatrixDim;
typedef std::pair<MatrixDim, floatt> MatrixElement;
typedef std::multimap<const math::Matrix*, MatrixElement> MatrixElements;

MatrixElements globalMatrixElementsSetRe;
MatrixElements globalMatrixElementsSetIm;
MatrixElements globalMatrixElementsGetRe;
MatrixElements globalMatrixElementsGetIm;
utils::sync::Mutex globalMutex;

namespace test {

bool isElement(const MatrixElements& matrixElements, const math::Matrix* matrix,
               uintt column, uintt row) {
  utils::sync::MutexLocker ml(globalMutex);
  std::pair<MatrixElements::const_iterator, MatrixElements::const_iterator>
      range = matrixElements.equal_range(matrix);
  for (MatrixElements::const_iterator it = range.first; it != range.second;
       ++it) {
    const uintt lcolumn = (*it).second.first.first;
    const uintt lrow = (*it).second.first.second;
    if (lcolumn == column && lrow == row) {
      return true;
    }
  }
  return false;
}

void addElement(MatrixElements& matrixElements, const math::Matrix* matrix,
                uintt column, uintt row, floatt value) {
  utils::sync::MutexLocker ml(globalMutex);
  matrixElements.insert(std::pair<const math::Matrix*, MatrixElement>(
      matrix, MatrixElement(MatrixDim(column, row), value)));
}

void removeElement(MatrixElements& matrixElements, const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  MatrixElements::iterator it = matrixElements.find(matrix);
  if (it != matrixElements.end()) {
    matrixElements.erase(it);
  }
}

void reset(const math::Matrix* matrix) {
  removeElement(globalMatrixElementsSetRe, matrix);
  removeElement(globalMatrixElementsSetIm, matrix);
  removeElement(globalMatrixElementsGetRe, matrix);
  removeElement(globalMatrixElementsGetIm, matrix);
}

void setRe(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsSetRe, matrix, column, row, value);
}

void setIm(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsSetIm, matrix, column, row, value);
}

void getRe(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsGetRe, matrix, column, row, value);
}

void getIm(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsGetIm, matrix, column, row, value);
}

bool wasSetRe(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsSetRe, matrix, column, row);
}

bool wasSetIm(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsSetIm, matrix, column, row);
}

bool wasGetRe(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsGetRe, matrix, column, row);
}

bool wasGetIm(const math::Matrix* matrix, uintt column, uintt row) {
  return isElement(globalMatrixElementsGetIm, matrix, column, row);
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

uintt getSetValuesCountRe(const math::Matrix* matrix) {
  return globalMatrixElementsSetRe.count(matrix);
}

uintt getSetValuesCountIm(const math::Matrix* matrix) {
  return globalMatrixElementsSetIm.count(matrix);
}

uintt getGetValuesCountRe(const math::Matrix* matrix) {
  return globalMatrixElementsGetRe.count(matrix);
}

uintt getGetValuesCountIm(const math::Matrix* matrix) {
  return globalMatrixElementsGetIm.count(matrix);
}

};
