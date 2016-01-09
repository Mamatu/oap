#include "MatrixTestAPI.h"

#include <map>
#include <stack>
#include <stdlib.h>
#include "ThreadUtils.h"

enum Type { Re, Im };

typedef std::pair<uintt, uintt> MatrixDim;
typedef std::pair<MatrixDim, floatt> MatrixElement;
typedef std::pair<const math::Matrix*, int> MatrixLevel;
typedef std::multimap<MatrixLevel, MatrixElement> MatrixElements;
typedef std::map<const math::Matrix*, MatrixLevel> MatrixLevels;

MatrixElements globalMatrixElementsSetRe;
MatrixElements globalMatrixElementsSetIm;
MatrixElements globalMatrixElementsGetRe;
MatrixElements globalMatrixElementsGetIm;
MatrixLevels globalCurrentLevel;
utils::sync::Mutex globalMutex;

namespace test {

inline MatrixLevel& getCurrentLevel(const math::Matrix* matrix) {
  if (globalCurrentLevel.count(matrix) == 0) {
    MatrixLevel matrixLevel(matrix, 0);
    globalCurrentLevel[matrix] = matrixLevel;
  }
  return globalCurrentLevel[matrix];
}

uintt calcColumn(const math::Matrix* matrix, uintt index) {
  return index % matrix->columns;
}

uintt calcRow(const math::Matrix* matrix, uintt index) {
  return index / matrix->columns;
}

bool isElement(const MatrixElements& matrixElements, const math::Matrix* matrix,
               uintt column, uintt row) {
  utils::sync::MutexLocker ml(globalMutex);
  std::pair<MatrixElements::const_iterator, MatrixElements::const_iterator>
      range = matrixElements.equal_range(getCurrentLevel(matrix));
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
  matrixElements.insert(std::pair<MatrixLevel, MatrixElement>(
      getCurrentLevel(matrix), MatrixElement(MatrixDim(column, row), value)));
}

void removeElement(MatrixElements& matrixElements, const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  MatrixElements::iterator it = matrixElements.find(getCurrentLevel(matrix));
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

void push(const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  globalCurrentLevel[matrix].second = globalCurrentLevel[matrix].second + 1;
}

void pop(const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  if (globalCurrentLevel[matrix].second > 0) {
    globalMatrixElementsSetRe.erase(globalCurrentLevel[matrix]);
    globalMatrixElementsSetIm.erase(globalCurrentLevel[matrix]);
    globalMatrixElementsGetRe.erase(globalCurrentLevel[matrix]);
    globalMatrixElementsGetIm.erase(globalCurrentLevel[matrix]);
    globalCurrentLevel[matrix].second = globalCurrentLevel[matrix].second - 1;
  }
}

uintt getStackLevels(const math::Matrix* matrix) {
  if (globalCurrentLevel.count(matrix) == 0) {
    return 0;
  }
  return globalCurrentLevel[matrix].second + 1;
}

void setRe(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsSetRe, matrix, column, row, value);
}

void setRe(const math::Matrix* matrix, uintt index, floatt value) {
  setRe(matrix, calcColumn(matrix, index), calcRow(matrix, index), value);
}

void setIm(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsSetIm, matrix, column, row, value);
}

void setIm(const math::Matrix* matrix, uintt index, floatt value) {
  setIm(matrix, calcColumn(matrix, index), calcRow(matrix, index), value);
}

void getRe(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsGetRe, matrix, column, row, value);
}

void getRe(const math::Matrix* matrix, uintt index, floatt value) {
  getRe(matrix, calcColumn(matrix, index), calcRow(matrix, index), value);
}

void getIm(const math::Matrix* matrix, uintt column, uintt row, floatt value) {
  addElement(globalMatrixElementsGetIm, matrix, column, row, value);
}

void getIm(const math::Matrix* matrix, uintt index, floatt value) {
  getIm(matrix, calcColumn(matrix, index), calcRow(matrix, index), value);
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
  utils::sync::MutexLocker ml(globalMutex);
  return globalMatrixElementsSetRe.count(getCurrentLevel(matrix));
}

uintt getSetValuesCountIm(const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  return globalMatrixElementsSetIm.count(getCurrentLevel(matrix));
}

uintt getGetValuesCountRe(const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  return globalMatrixElementsGetRe.count(getCurrentLevel(matrix));
}

uintt getGetValuesCountIm(const math::Matrix* matrix) {
  utils::sync::MutexLocker ml(globalMutex);
  return globalMatrixElementsGetIm.count(getCurrentLevel(matrix));
}
};
