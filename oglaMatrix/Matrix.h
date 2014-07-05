/* 
 * File:   Matrix.h
 * Author: mmatula
 *
 * Created on November 29, 2013, 6:29 PM
 */

#ifndef OGLA_MATRIX_H
#define	OGLA_MATRIX_H
#include "Math.h"

namespace math {

    /**
     * Columns orientation 
     */
    struct Matrix {
        floatt* reValues;
        floatt* imValues;
        uintt columns;
        uintt rows;
    };
}
#endif

