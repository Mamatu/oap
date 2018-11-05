#ifndef MATRIXINFO_H
#define MATRIXINFO_H

#include "Matrix.h"
#include <sstream>
#include <utility>
#include <cstddef>

#include <limits>

namespace math
{

class MatrixInfo
{
 int toInt (bool b) const
 {
   return b ? 1 : 0;
 }

 public:
  enum Units
  {
    B = 0,
    KB = 1,
    MB,
    GB
  };

  inline MatrixInfo() : isRe(false), isIm(false) {
    m_matrixDim.columns = 0;
    m_matrixDim.rows = 0;
  }

  inline MatrixInfo(bool _isRe, bool _isIm, uintt _columns, uintt _rows)
      : isRe(_isRe), isIm(_isIm) {
    m_matrixDim.columns = _columns;
    m_matrixDim.rows = _rows;
  }

  explicit inline MatrixInfo(math::Matrix* hostMatrix) {
    isRe = hostMatrix->reValues != NULL;
    isIm = hostMatrix->imValues != NULL;

    m_matrixDim.columns = hostMatrix->columns;
    m_matrixDim.rows = hostMatrix->rows;
  }

  bool isInitialized() const
  {
    return !(!isRe && !isIm);
  }

  void deinitialize ()
  {
    isRe = false;
    isIm = false;
  }

  bool operator==(const MatrixInfo& mInfo) const
  {
    return isRe == mInfo.isRe && isIm == mInfo.isIm &&
           m_matrixDim.columns == mInfo.m_matrixDim.columns && m_matrixDim.rows == mInfo.m_matrixDim.rows;
  }

  bool operator!=(const MatrixInfo& minfo) const
  {
    return !(*this == minfo);
  }

  operator bool() const
  {
    return isInitialized ();
  }

  std::string toString() const
  {
    std::stringstream stream;
    std::pair<size_t, Units> size = getSize();
    stream << "(" << isRe << ", " << isIm << ", " << m_matrixDim.columns << ", " << m_matrixDim.rows << ", " << size.first << " " << toString(size.second) << ")";
    return stream.str();
  }

  std::string toString(Units units) const
  {
    if (units == B) { return "B"; }
    if (units == KB) { return "KiB"; }
    if (units == MB) { return "MiB"; }
    if (units == GB) { return "GiB"; }
    return "";
  }

  std::pair<size_t, Units> getSize () const
  {
    Units units = B;
    size_t size = getSizeInBytes ();

    uintt tsize = getSizeInKB ();
    if (tsize > 0)
    {
      size = tsize; units = KB;
    }

    tsize = getSizeInMB ();
    if (tsize > 0)
    {
      size = tsize; units = MB;
    }

    tsize = getSizeInGB ();
    if (tsize > 0)
    {
      size = tsize; units = GB;
    }

    return std::make_pair(size, units);
  }

  size_t getSizeInBytes () const
  {
    size_t size1 = m_matrixDim.columns * m_matrixDim.rows;
    size_t size = (toInt (isRe) + toInt (isIm)) * size1 * sizeof(floatt);
    return size;
  }

  size_t getSizeInKB () const
  {
    return getSizeInBytes () / 1024;
  }

  size_t getSizeInMB () const
  {
    return getSizeInKB () / (1024);
  }

  size_t getSizeInGB () const
  {
    return getSizeInMB () / (1024);
  }

  math::MatrixDim m_matrixDim;
  bool isRe;
  bool isIm;
};

}

#endif  // MATRIXINFO_H
