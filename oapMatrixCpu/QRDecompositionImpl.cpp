/*
 * Copyright 2016 - 2018 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#include "MathOperationsCpu.h"
#include "ThreadData.h"
#include <math.h>
namespace math {

inline void switchPointer(math::Matrix*& a, math::Matrix*& b) {
    math::Matrix* temp = b;
    b = a;
    a = temp;
}

inline void QRDecompositionCpu::prepareGMatrix(math::Matrix* A,
    uintt column, uintt row,
    math::Matrix* G) {
    oap::host::SetIdentityMatrix(G);
    oap::host::SetIdentityMatrix(m_GT);
    floatt reg = 0;
    floatt img = 0;
    floatt ref = 0;
    floatt imf = 0;
    if (A->reValues) {
        reg = A->reValues[column + row * A->columns];
        ref = A->reValues[column + column * A->columns];
    }
    if (A->imValues) {
        img = A->imValues[column + row * A->columns];
        imf = A->imValues[column + column * A->columns];
    }
    floatt r = sqrt(ref * ref + reg * reg + img * img + imf * imf);
    floatt lf = sqrt(ref * ref + imf * imf);
    floatt sign = 1;
    floatt isign = 0;
    if (fabs(ref) >= MATH_VALUE_LIMIT || fabs(imf) >= MATH_VALUE_LIMIT) {
        sign = ref / lf;
        isign = imf / lf;
    }
    floatt s = (sign * reg + img * isign) / r;
    floatt is = (isign * reg - img * sign) / r;
    floatt c = lf / r;
    floatt ic = 0;
    if (G->reValues) {
        G->reValues[column + row * A->columns] = -s;
        G->reValues[column + column * A->columns] = c;
        G->reValues[row + row * A->columns] = c;
        G->reValues[row + column * A->columns] = s;
    }
    if (G->imValues) {
        G->imValues[column + row * A->columns] = is;
        G->imValues[column + column * A->columns] = ic;
        G->imValues[row + row * A->columns] = ic;
        G->imValues[row + column * A->columns] = is;
    }
}

void QRDecompositionCpu::execute() {
    dotProduct.setThreadsCount(this->m_threadsCount);
    transpose.setThreadsCount(this->m_threadsCount);
    math::Matrix* A = this->m_matrix;
    math::Matrix* Q = this->m_output1;
    math::Matrix* R = this->m_output2;

    math::Matrix* tR1 = NULL;
    math::Matrix* tQ1 = NULL;

    if (m_R1 != NULL && m_R1->rows != R->rows &&
        m_R1->columns != R->columns) {
oap::host::DeleteMatrix(m_R1);
oap::host::DeleteMatrix(m_Q1);
oap::host::DeleteMatrix(m_G);
oap::host::DeleteMatrix(m_GT);
        m_R1 = NULL;
        m_Q1 = NULL;
        m_G = NULL;
        m_GT = NULL;
    }
    if (m_R1 == NULL) {
        m_R1 = oap::host::NewMatrix(A);
        m_Q1 = oap::host::NewMatrix(A);
        m_G = oap::host::NewMatrix(A);
        m_GT = oap::host::NewMatrix(A);
    }

    tR1 = m_R1;
    tQ1 = m_Q1;

    oap::host::CopyMatrix(m_R1, A);
    oap::host::SetIdentityMatrix(m_Q1);
    for (uintt fa = 0; fa < A->columns - 1; ++fa) {
        for (uintt fb = A->rows - 1; fb > fa; --fb) {
            floatt rev = m_R1->reValues[fa + fb * m_R1->columns];
            floatt imv = 0;
            if (m_R1->imValues) {
                imv = m_R1->imValues[fa + fb * m_R1->columns];
            }
            if ((fabs(rev) < MATH_VALUE_LIMIT &&
                fabs(imv) < MATH_VALUE_LIMIT) == false) {
                prepareGMatrix(m_R1, fa, fb, m_G);
                
                dotProduct.setMatrix2(m_R1);
                dotProduct.setMatrix1(m_G);
                dotProduct.setOutputMatrix(R);
                dotProduct.start();

                transpose.setMatrix(m_G);
                transpose.setOutputMatrix(m_GT);
                transpose.start();

                dotProduct.setMatrix2(m_GT);
                dotProduct.setMatrix1(m_Q1);
                dotProduct.setOutputMatrix(Q);
                dotProduct.start();

                switchPointer(m_R1, R);
                switchPointer(m_Q1, Q);
            }
        }
    }
    if (this->m_output1 != m_Q1) {
        oap::host::CopyMatrix(this->m_output1, m_Q1);
    }
    if (this->m_output2 != m_R1) {
        oap::host::CopyMatrix(this->m_output2, m_R1);
    }
    m_R1 = tR1;
    m_Q1 = tQ1;
}

QRDecompositionCpu::QRDecompositionCpu() :
    IQRDecomposition(),
    m_R1(NULL), m_Q1(NULL), m_G(NULL), m_GT(NULL) {
}

QRDecompositionCpu::~QRDecompositionCpu() {
    if (m_R1) {
oap::host::DeleteMatrix(m_R1);
oap::host::DeleteMatrix(m_Q1);
oap::host::DeleteMatrix(m_G);
oap::host::DeleteMatrix(m_GT);
    }
}
}
