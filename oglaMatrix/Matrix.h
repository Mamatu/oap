/*
 * File:   Matrix.h
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#ifndef OGLA_MATRIX_H
#define OGLA_MATRIX_H

#include <assert.h>
#include "Math.h"

#define DEBUG

namespace math {

struct MatrixDim {
  uintt columns;
  uintt rows;
};

/**
 * Columns orientation
 */
struct Matrix {
  uintt realColumns;
  uintt realRows;
  floatt* reValues;
  floatt* imValues;
  uintt columns;
  uintt rows;
};
}

#if RELEASE

#define SetRe(m, c, r, v) m->reValues[c + r * m->columns] = v;

#define GetRe(m, c, r, v) m->reValues[c + r * m->columns];

#define SetIm(m, c, r, v) m->imValues[c + r * m->columns] = v;

#define GetIm(m, c, r, v) m->imValues[c + r * m->columns];

#define Reset(m)

#elif CUDATEST

namespace test {
void reset(const math::Matrix* matrix);
void setRe(const math::Matrix* matrix, uintt column, uintt row, floatt value);
void setIm(const math::Matrix* matrix, uintt column, uintt row, floatt value);
bool wasSetRe(const math::Matrix* matrix, uintt column, uintt row);
bool wasSetIm(const math::Matrix* matrix, uintt column, uintt row);
bool wasSetRangeRe(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasSetRangeIm(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasSetAllRe(const math::Matrix* matrix);
bool wasSetAllIm(const math::Matrix* matrix);
floatt getRe(const math::Matrix* matrix, uintt column, uintt row, floatt value);
floatt getIm(const math::Matrix* matrix, uintt column, uintt row, floatt value);
bool wasGetRe(const math::Matrix* matrix, uintt column, uintt row);
bool wasGetIm(const math::Matrix* matrix, uintt column, uintt row);
bool wasGetRangeRe(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasGetRangeIm(const math::Matrix* matrix, uintt bcolumn, uintt ecolumn,
                   uintt brow, uintt erow);
bool wasGetAllRe(const math::Matrix* matrix);
bool wasGetAllIm(const math::Matrix* matrix);
};

#define SetRe(m, c, r, v)              \
  m->reValues[c + r * m->columns] = v; \
  test::setRe(m, c, r, v);             \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define GetRe(m, c, r)                                   \
  m->reValues[c + r * m->columns];                       \
  test::getRe(m, c, r, m->reValues[c + r * m->columns]); \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define SetIm(m, c, r, v)              \
  m->imValues[c + r * m->columns] = v; \
  test::setIm(m, c, r, v);             \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define GetIm(m, c, r)                                   \
  m->imValues[c + r * m->columns];                       \
  test::getIm(m, c, r, m->imValues[c + r * m->columns]); \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define Reset(m) test::reset(m);

#else

#define SetRe(m, c, r, v)              \
  m->reValues[c + r * m->columns] = v; \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define GetRe(m, c, r)             \
  m->reValues[c + r * m->columns]; \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define SetIm(m, c, r, v)              \
  m->imValues[c + r * m->columns] = v; \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define GetIm(m, c, r)             \
  m->imValues[c + r * m->columns]; \
  assert(c + r * m->columns < m->realRows * m->realColumns);

#define Reset(m)

#endif

#endif
