#ifndef OAP_RANGE_H
#define OAP_RANGE_H

#include "Math.h"

namespace matrixUtils
{

class Range {
 protected:
  uintt m_bcolumn;
  uintt m_columns;
  uintt m_brow;
  uintt m_rows;

 public:
  Range(uintt bcolumn, uintt columns, uintt brow, uintt rows);

  virtual ~Range();

  virtual uintt getBColumn() const;
  virtual uintt getEColumn() const;
  virtual uintt getColumns() const;

  virtual uintt getBRow() const;
  virtual uintt getERow() const;
  virtual uintt getRows() const;
};
}

#endif
