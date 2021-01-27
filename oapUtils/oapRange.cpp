#include "oapRange.h"

namespace matrixUtils
{

Range::Range(uintt bcolumn, uintt columns, uintt brow, uintt rows)
    : m_bcolumn(bcolumn), m_columns(columns), m_brow(brow), m_rows(rows) {}

Range::~Range() {}

uintt Range::getBColumn() const { return m_bcolumn; }
uintt Range::getEColumn() const { return m_bcolumn + m_columns; }
uintt Range::getColumns() const { return m_columns; }

uintt Range::getBRow() const { return m_brow; }
uintt Range::getERow() const { return m_brow + m_rows; }
uintt Range::getRows() const { return m_rows; }

}
