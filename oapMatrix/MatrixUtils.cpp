/*
 * Copyright 2016 - 2019 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "MatrixUtils.h"
#include <algorithm>
#include <locale>
#include <math.h>
#include <stdlib.h>
#include <DebugLogs.h>

namespace matrixUtils {

const char* ID_COLUMNS = "columns";
const char* ID_ROWS = "rows";
const char* ID_LENGTH = "length";

Range::Range(uintt bcolumn, uintt columns, uintt brow, uintt rows)
    : m_bcolumn(bcolumn), m_columns(columns), m_brow(brow), m_rows(rows) {}

Range::~Range() {}

uintt Range::getBColumn() const { return m_bcolumn; }
uintt Range::getEColumn() const { return m_bcolumn + m_columns; }
uintt Range::getColumns() const { return m_columns; }

uintt Range::getBRow() const { return m_brow; }
uintt Range::getERow() const { return m_brow + m_rows; }
uintt Range::getRows() const { return m_rows; }

MatrixRange::MatrixRange(const math::Matrix* matrix)
    : Range(0, matrix->columns, 0, matrix->rows), m_matrix(matrix) {}

MatrixRange::MatrixRange(const math::Matrix* matrix, const Range& range)
    : Range(range), m_matrix(matrix) {}

MatrixRange::MatrixRange(const math::Matrix* matrix, uintt bcolumn,
                         uintt columns, uintt brow, uintt rows)
    : Range(bcolumn, columns, brow, rows), m_matrix(matrix) {}

MatrixRange::~MatrixRange() {}

bool MatrixRange::isReValues() const { return m_matrix->reValues != NULL; }

bool MatrixRange::isImValues() const { return m_matrix->imValues != NULL; }

const math::Matrix* MatrixRange::getMatrix() const { return m_matrix; }

uintt MatrixRange::getEColumn() const {
  return std::min(m_bcolumn + m_columns, m_matrix->columns);
}

uintt MatrixRange::getERow() const {
  return std::min(m_brow + m_rows, m_matrix->rows);
}

void MatrixRange::getReSubArrays(SubArrays<floatt>& subArrays) const {
  getSubArrays(subArrays, m_matrix->reValues, m_matrix);
}

void MatrixRange::getImSubArrays(SubArrays<floatt>& subArrays) const {
  getSubArrays(subArrays, m_matrix->imValues, m_matrix);
}

void MatrixRange::getSubArrays(SubArrays<floatt>& subArrays, floatt* array,
                               const math::Matrix* matrix) const {
  for (uintt fa = m_brow; fa < m_rows; ++fa) {
    uintt bindex = m_bcolumn + (m_brow + fa) * matrix->columns;
    subArrays.push_back(std::make_pair(&array[bindex], m_columns));
  }
}

Section::Section ()
{}

Section::Section (const std::string& _separator, size_t _length) : separator (_separator), length (_length)
{}

PrintArgs::PrintArgs ()
{}

PrintArgs::PrintArgs (const std::string& _pretext, const std::string& _posttext, floatt _zrr, bool _repeats, const std::string& _sectionSeparator, size_t _sectionLength):
  pretext(_pretext), posttext(_posttext), zrr(_zrr), repeats(_repeats), section(_sectionSeparator, _sectionLength)
{}

PrintArgs::PrintArgs (floatt _zrr, bool _repeats):
  zrr(_zrr), repeats(_repeats)
{}

PrintArgs::PrintArgs (floatt _zrr, bool _repeats, const std::string& _sectionSeparator, size_t _sectionLength):
  zrr(_zrr), repeats(_repeats), section (_sectionSeparator, _sectionLength)
{}

PrintArgs::PrintArgs (const std::string& _pretext, const std::string& _posttext) : pretext(_pretext), posttext(_posttext)
{}

PrintArgs::PrintArgs (const std::string& _pretext) : pretext(_pretext)
{}

PrintArgs::PrintArgs (const char* _pretext) : pretext(_pretext)
{}

void PrintArgs::setReImSeparator (const std::string& _postRe, const std::string& _preIm)
{
  postRe = _postRe;
  preIm = _preIm;
}

}
