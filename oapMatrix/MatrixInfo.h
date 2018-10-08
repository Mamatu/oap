#ifndef MATRIXINFO_H
#define MATRIXINFO_H

#include "Matrix.h"
#include <cstddef>

namespace math {

class MatrixInfo {
 public:
  inline MatrixInfo() : isRe(false), isIm(false) {
    m_matrixDim.columns = 0;
    m_matrixDim.rows = 0;
  }

  inline MatrixInfo(bool _isRe, bool _isIm, uintt _columns, uintt _rows)
      : isRe(_isRe), isIm(_isIm) {
    m_matrixDim.columns = _columns;
    m_matrixDim.rows = _rows;
  }

  inline MatrixInfo(math::Matrix* hostMatrix) {
    isRe = hostMatrix->reValues != NULL;
    isIm = hostMatrix->imValues != NULL;

    m_matrixDim.columns = hostMatrix->columns;
    m_matrixDim.rows = hostMatrix->rows;
  }

  bool isInitialized() const
  {
    return !(!isRe && !isIm);
  }

  bool operator==(const MatrixInfo& mInfo)
  {
    return isRe != mInfo.isRe || isIm != mInfo.isIm || 
           m_matrixDim.columns != mInfo.m_matrixDim.columns || m_matrixDim.rows != mInfo.m_matrixDim.rows;
  }

  bool operator!=(const MatrixInfo& minfo)
  {
    return !(*this == minfo);
  }

  math::MatrixDim m_matrixDim;
  bool isRe;
  bool isIm;
};
}

#endif  // MATRIXINFO_H
