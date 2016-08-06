/*
 * File:   Matrix.cpp
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#ifndef OGLA_MATRIX_TEST_API_H
#define OGLA_MATRIX_TEST_API_H

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
