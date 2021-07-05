#ifndef OAP_MATRIX_INFO_H
#define OAP_MATRIX_INFO_H

#include "Matrix.hpp"
#include <sstream>
#include <utility>
#include <cstddef>

#include <array>
#include <limits>
#include <cmath>

namespace math
{

class MatrixInfo
{
 int toInt (bool b) const;

 public:
  enum Units
  {
    B = 0,
    KB = 1,
    MB,
    GB
  };

  MatrixInfo();

  MatrixInfo(bool _isRe, bool _isIm, uintt _columns, uintt _rows);

  explicit MatrixInfo (const math::ComplexMatrix* hostMatrix);
  explicit MatrixInfo (const math::ComplexMatrix& hostMatrix);

  bool isInitialized() const;

  void deinitialize ();

  bool operator== (const MatrixInfo& mInfo) const;
  bool operator!= (const MatrixInfo& mInfo) const;
  bool operator< (const MatrixInfo& mInfo) const;

  operator bool() const;

  std::string toString() const;

  std::string toString(Units units) const;

  std::array<size_t, 4> getSizeInBuffers() const;

  std::pair<size_t, Units> getSize () const;

  uintt columns() const;
  uintt rows() const;

  /**
   * Length of array in c++ meaning (columns () * rows ())
   */
  uintt length() const
  {
    return columns() * rows();
  }

  math::MatrixDim m_matrixDim;
  bool isRe;
  bool isIm;
};
}

namespace oap
{
namespace math
{
using MatrixInfo = ::math::MatrixInfo;
}
}

namespace std
{
  std::string to_string (const math::MatrixInfo& minfo);
}

#endif  // MATRIXINFO_H
