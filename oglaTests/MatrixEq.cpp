/* 
 * File:   matrixEq.cpp
 * Author: mmatula
 * 
 * Created on December 23, 2014, 10:30 AM
 */

#include "MatrixEq.h"

bool operator==(const math::Matrix& m1, const math::Matrix& m2) {
    if (m1.columns != m2.columns || m1.rows != m2.rows) {
        return false;
    }
    for (uintt fa = 0; fa < m1.columns; ++fa) {
        for (uintt fb = 0; fb < m1.rows; ++fb) {
            if (m1.reValues[fa + fb * m1.columns] != m2.reValues[fa + fb * m2.columns]) {
                return false;
            }
        }
    }
    return true;
}

bool IsEqual(const MatrixEx& matrixEx, const uintt* buffer) {
    if (matrixEx.bcolumn == buffer[0]
        && matrixEx.ecolumn == buffer[1]
        && matrixEx.brow == buffer[2]
        && matrixEx.erow == buffer[3]
        && matrixEx.boffset == buffer[4]
        && matrixEx.eoffset == buffer[5]) {
        return true;
    }
    return false;
}

