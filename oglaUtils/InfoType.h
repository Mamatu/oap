#ifndef OAP_INFOTYPE
#define OAP_INFOTYPE

#include "MatrixUtils.h"
#include <limits>

class InfoType : public matrixUtils::Range {
 public:
  enum Type {
    ELEMENTS = 1 << 0,
    MEAN = 1 << 1,
    LARGEST_DIFF = 1 << 2,
    SMALLEST_DIFF = 1 << 3
  };

 private:
  Type m_type;

 public:
  inline InfoType(uintt bcolumn, uintt columns, uintt brow, uintt rows,
                  InfoType::Type type)
      : matrixUtils::Range(bcolumn, columns, brow, rows), m_type(type) {}

  inline InfoType()
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(ELEMENTS) {}

  inline InfoType(InfoType::Type type)
      : matrixUtils::Range(0, std::numeric_limits<uintt>::max(), 0,
                           std::numeric_limits<uintt>::max()),
        m_type(type) {}

  inline InfoType::Type getInfo() const { return m_type; }
};

#endif  // INFOTYPE
