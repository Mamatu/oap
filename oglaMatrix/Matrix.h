/* 
 * File:   Matrix.h
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#ifndef OGLA_MATRIX_H
#define	OGLA_MATRIX_H

#include <assert.h>
#include "Math.h"

#define DEBUG

namespace math {

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

#ifdef DEBUG

#define SetRe(m, c, offset, r, v) m->reValues[c + r * offset] = v;\
assert(c + r * offset < m->realRows * m->realColumns);

#define SetReIndex(m, index, v) m->reValues[index] = v;\
assert(index < m->realRows * m->realColumns);

#define SetIm(m, c, offset, r, v) m->imValues[c + r * offset] = v;\
assert(c + r * offset , m->realRows * m->realColumns);

#define SetImIndex(m, index, v) m->imValues[index] = v;\
assert(index < m->realRows * m->realColumns);

#else

#define SetRe(m, c, offset, r, v) m->reValues[c + r * offset] = v;\

#define SetReIndex(m, index, v) m->reValues[index] = v;\

#define SetIm(m, c, offset, r, v) m->imValues[c + r * offset] = v;\

#define SetImIndex(m, index, v) m->imValues[index] = v;\

#endif

#endif

