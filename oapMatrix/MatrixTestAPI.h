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



#ifndef OAP_MATRIX_TEST_API_H
#define OAP_MATRIX_TEST_API_H

#include "Matrix.h"

namespace test {
void reset(const math::Matrix* matrix);
void push(const math::Matrix* matrix);
void pop(const math::Matrix* matrix);
uintt getStackLevels(const math::Matrix* matrix);
void setRe(const math::Matrix* matrix, uintt column, uintt row, floatt value);
void setRe(const math::Matrix* matrix, uintt index, floatt value);
void setIm(const math::Matrix* matrix, uintt column, uintt row, floatt value);
void setIm(const math::Matrix* matrix, uintt index, floatt value);
bool wasSetRe(const math::Matrix* matrix, uintt column, uintt row);
bool wasSetIm(const math::Matrix* matrix, uintt column, uintt row);

bool wasSetRangeRe(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasSetRangeIm(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasSetAllRe(const math::Matrix* matrix);
bool wasSetAllIm(const math::Matrix* matrix);
void getRe(const math::Matrix* matrix, uintt column, uintt row, floatt value);
void getRe(const math::Matrix* matrix, uintt index, floatt value);
void getIm(const math::Matrix* matrix, uintt column, uintt row, floatt value);
void getIm(const math::Matrix* matrix, uintt index, floatt value);
bool wasGetRe(const math::Matrix* matrix, uintt column, uintt row);
bool wasGetIm(const math::Matrix* matrix, uintt column, uintt row);
bool wasGetRangeRe(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasGetRangeIm(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasGetAllRe(const math::Matrix* matrix);
bool wasGetAllIm(const math::Matrix* matrix);
uintt getSetValuesCountRe(const math::Matrix* matrix);
uintt getSetValuesCountIm(const math::Matrix* matrix);
uintt getGetValuesCountRe(const math::Matrix* matrix);
uintt getGetValuesCountIm(const math::Matrix* matrix);
};

#endif
