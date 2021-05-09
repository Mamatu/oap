/*
 * Copyright 2016 - 2021 Marcin Matula
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

inline void switchPointer(math::ComplexMatrix*& a, math::ComplexMatrix*& b) {
    math::ComplexMatrix* temp = b;
    b = a;
    a = temp;
}

inline void QRDecompositionCpu::prepareGMatrix(math::ComplexMatrix* A,
    uintt column, uintt row,
    math::ComplexMatrix* G) {
    oap::host::SetIdentityMatrix(G);
    oap::host::SetIdentityMatrix(m_GT);
    floatt reg = 0;
    floatt img = 0;
    floatt ref = 0;
    floatt imf = 0;
    if (gReValues (A)) {
        reg = gReValues (A)[column + row * gColumns (A)];
        ref = gReValues (A)[column + column * gColumns (A)];
    }
    if (gImValues (A)) {
        img = gImValues (A)[column + row * gColumns (A)];
        imf = gImValues (A)[column + column * gColumns (A)];
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
    if (gReValues (G)) {
        gReValues (G)[column + row * gColumns (A)] = -s;
        gReValues (G)[column + column * gColumns (A)] = c;
        gReValues (G)[row + row * gColumns (A)] = c;
        gReValues (G)[row + column * gColumns (A)] = s;
    }
    if (gImValues (G)) {
        gImValues (G)[column + row * gColumns (A)] = is;
        gImValues (G)[column + column * gColumns (A)] = ic;
        gImValues (G)[row + row * gColumns (A)] = ic;
        gImValues (G)[row + column * gColumns (A)] = is;
    }
}

void QRDecompositionCpu::execute() {
    dotProduct.setThreadsCount(this->m_threadsCount);
    transpose.setThreadsCount(this->m_threadsCount);
    math::ComplexMatrix* A = this->m_matrix;
    math::ComplexMatrix* Q = this->m_output1;
    math::ComplexMatrix* R = this->m_output2;

    math::ComplexMatrix* tR1 = NULL;
    math::ComplexMatrix* tQ1 = NULL;

    if (m_R1 != NULL && gRows (m_R1) != gRows (R) &&
        gColumns (m_R1) != gColumns (R)) {
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
        m_R1 = oap::host::NewComplexMatrixRef (A);
        m_Q1 = oap::host::NewComplexMatrixRef (A);
        m_G = oap::host::NewComplexMatrixRef (A);
        m_GT = oap::host::NewComplexMatrixRef (A);
    }

    tR1 = m_R1;
    tQ1 = m_Q1;

    oap::host::CopyMatrix(m_R1, A);
    oap::host::SetIdentityMatrix(m_Q1);
    for (uintt fa = 0; fa < gColumns (A) - 1; ++fa) {
        for (uintt fb = gRows (A) - 1; fb > fa; --fb) {
            floatt rev = gReValues (m_R1)[fa + fb * gColumns (m_R1)];
            floatt imv = 0;
            if (gImValues (m_R1)) {
                imv = gImValues (m_R1)[fa + fb * gColumns (m_R1)];
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
