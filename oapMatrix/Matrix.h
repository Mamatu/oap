/*
 * File:   Matrix.h
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#ifndef OGLA_MATRIX_H
#define OGLA_MATRIX_H

#include "Math.h"

//#define DEBUG

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


#endif
