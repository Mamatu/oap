/* 
 * File:   DeviceIram.h
 * Author: mmatula
 *
 * Created on July 10, 2014, 10:13 PM
 */

#ifndef DEVICEIRAM_H
#define	DEVICEIRAM_H

#include "Device.h"
//#if 0

extern "C" __device__
bool IsTriangular(MatrixStructure* matrix, uintt count) {
    uintt index = 0;
    for (uintt fa = 0; fa < matrix->m_matrix->columns - 1; ++fa) {
        floatt revalue =
                matrix->m_matrix->
                reValues[fa + matrix->m_matrix->columns * (fa + 1)];
        floatt imvalue = 0;
        if (matrix->m_matrix->imValues) {
            imvalue = matrix->m_matrix->
                    imValues[fa + matrix->m_matrix->columns * (fa + 1)];
        }
        if ((-0.00000000001 < revalue && revalue < 0.00000000001) &&
                (-0.00000000001 < imvalue && imvalue < 0.00000000001)) {
            index++;
        }
    }
    if (index >= count) {
        return true;
    } else {
        return false;
    }
}

extern "C" __device__
void calculateH(
        MatrixStructure* H1,
        MatrixStructure* Q1,
        MatrixStructure* R1,
        MatrixStructure* QJ,
        MatrixStructure* Q,
        uintt threadIndexX, uintt threadIndexY) {
    for (uintt fa = 0; IsTriangular(H1, H1->m_matrix->columns - 1) == false; ++fa) {
        multiplyMatrices(H1, R1, Q1, threadIndexX, threadIndexY);
        multiplyMatrices(Q, QJ, Q1, threadIndexX, threadIndexY);
        if (threadIndexX == 1 && threadIndexY == 1) {
            switchPointer(Q, QJ);
        }
    }
}

extern "C" __device__
bool ExecuteArnoldiFactorization(bool init, intt initj) {

}

#endif	/* DEVICEIRAM_H */

