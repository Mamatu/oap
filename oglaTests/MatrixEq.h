/* 
 * File:   matrixEq.h
 * Author: mmatula
 *
 * Created on December 23, 2014, 10:30 AM
 */

#ifndef MATRIXEQ_H
#define	MATRIXEQ_H

#include "Matrix.h"
#include "MatrixEx.h"
bool operator==(const math::Matrix& m1, const math::Matrix& m2);

bool IsEqual(const MatrixEx& matrixEx, const uintt* buffer);

#endif	/* MATRIXEQ_H */

