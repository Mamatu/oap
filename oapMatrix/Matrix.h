#ifndef OAP_MATRIX_H
#define OAP_MATRIX_H

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
