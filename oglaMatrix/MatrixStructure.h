/* 
 * File:   MatrixStructure.h
 * Author: mmatula
 *
 * Created on May 24, 2014, 8:36 AM
 */

#ifndef OGLA_MATRIXSTRUCTURE_H
#define	OGLA_MATRIXSTRUCTURE_H

#include "Matrix.h"

struct MatrixStructure {
    math::Matrix* m_matrix;
    uintt m_beginColumn;
    uintt m_beginRow;
    uintt m_subcolumns;
    uintt m_subrows;
};

#endif	/* MATRIXSTRUCTURE_H */

