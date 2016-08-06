#ifndef OAP_INFOTYPE
#define OAP_INFOTYPE

#include "MatrixUtils.h"
#include <limits>

class InfoType : public matrixUtils::Range {
 public:
  static int ELEMENTS;
  static int MEAN;
  static int LARGEST_DIFF;
  static int SMALLEST_DIFF;

 private:
  int m_type;

 public:
  inline InfoType(uintt bcolumn, uintt columns, uintt brow, uintt rows,
                  int type)
      : matrixUtils::Range(bcolumn, columns, brow, rows), m_type(type) {}

  inline InfoType()
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(ELEMENTS) {}

  inline InfoType(int type)
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(type) {}

  inline int getInfo() const { return m_type; }
};

#endif  // INFOTYPE
