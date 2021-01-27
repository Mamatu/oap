#include "MatrixInfo.h"
#include "MatrixAPI.h"

namespace math
{

int MatrixInfo::toInt (bool b) const
{
  return b ? 1 : 0;
}


MatrixInfo::MatrixInfo() : isRe(false), isIm(false)
{
  m_matrixDim.columns = 0;
  m_matrixDim.rows = 0;
}

MatrixInfo::MatrixInfo(bool _isRe, bool _isIm, uintt _columns, uintt _rows) : isRe(_isRe), isIm(_isIm)
{
  m_matrixDim.columns = _columns;
  m_matrixDim.rows = _rows;
}

MatrixInfo::MatrixInfo (const math::Matrix* hostMatrix)
{
  isRe = gReValues (hostMatrix) != NULL;
  isIm = gImValues (hostMatrix) != NULL;

  m_matrixDim.columns = gColumns (hostMatrix);
  m_matrixDim.rows = gRows (hostMatrix);
}

MatrixInfo::MatrixInfo (const math::Matrix& hostMatrix) : MatrixInfo (&hostMatrix)
{
  //empty
}

bool MatrixInfo::isInitialized() const
{
  return !(!isRe && !isIm);
}

void MatrixInfo::deinitialize ()
{
  isRe = false;
  isIm = false;
}

bool MatrixInfo::operator==(const MatrixInfo& mInfo) const
{
  return isRe == mInfo.isRe && isIm == mInfo.isIm &&
         m_matrixDim.columns == mInfo.m_matrixDim.columns && m_matrixDim.rows == mInfo.m_matrixDim.rows;
}

bool MatrixInfo::operator!=(const MatrixInfo& mInfo) const
{
  return !(*this == mInfo);
}

bool MatrixInfo::operator< (const MatrixInfo& minfo) const
{
  if (this->isRe < minfo.isRe)
  {
    return true;
  }
  if (this->isIm < minfo.isIm)
  {
    return true;
  }
  if (this->columns() < minfo.columns())
  {
    return true;
  }
  if (this->rows() < minfo.rows())
  {
    return true;
  }
  return false;
}

MatrixInfo::operator bool() const
{
  return isInitialized ();
}

std::string MatrixInfo::toString() const
{
  std::stringstream stream;
  std::pair<size_t, Units> size = getSize();
  stream << "(" << isRe << ", " << isIm << ", " << m_matrixDim.columns << ", " << m_matrixDim.rows << ", " << size.first << " " << toString(size.second) << ")";
  return stream.str();
}

std::string MatrixInfo::toString(Units units) const
{
  if (units == B) { return "B"; }
  if (units == KB) { return "KiB"; }
  if (units == MB) { return "MiB"; }
  if (units == GB) { return "GiB"; }
  return "";
}

std::array<size_t, 4> MatrixInfo::getSizeInBuffers() const
{
  std::array<size_t, 4> sizes {0, 0, 0, 0};

  size_t s1 = 0;
  size_t s2 = 0;

  if (m_matrixDim.columns <= m_matrixDim.rows)
  {
    s1 = m_matrixDim.rows;
    s2 = m_matrixDim.columns;
  }
  else
  {
    s1 = m_matrixDim.columns;
    s2 = m_matrixDim.rows;
  }

  for (int idx = 3; idx >= 0; --idx)
  {
    floatt size = static_cast<floatt>(s1) / pow(1024, idx);
    size = size * s2;
    sizes[idx] = size * (toInt (isRe) + toInt (isIm)) * sizeof(floatt);
  }
  return sizes;
}

std::pair<size_t, MatrixInfo::Units> MatrixInfo::getSize () const
{
  std::array<size_t, 4> sizes = getSizeInBuffers ();

  for (int idx = 3; idx >= 0; --idx)
  {
    Units units = static_cast<Units>(idx);
    size_t rsize = sizes[units];
    if (rsize > 0)
    {
      return std::make_pair(rsize, units);
    }
  }
  return std::make_pair(0, KB);
}

uintt MatrixInfo::columns() const
{
  return m_matrixDim.columns;
}

uintt MatrixInfo::rows() const
{
  return m_matrixDim.rows;
}

}

namespace std
{
  std::string to_string (const math::MatrixInfo& minfo)
  {
    return minfo.toString ();
  }
}
