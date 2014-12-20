#include "MathOperationsCpu.h"
#include "Internal.h"
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
        m_module->getMatrixUtils()->setIdentityMatrix(G);
        m_module->getMatrixUtils()->setIdentityMatrix(m_GT);
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
            m_module->getMatrixAllocator()->deleteMatrix(m_R1);
            m_module->getMatrixAllocator()->deleteMatrix(m_Q1);
            m_module->getMatrixAllocator()->deleteMatrix(m_G);
            m_module->getMatrixAllocator()->deleteMatrix(m_GT);
            m_R1 = NULL;
            m_Q1 = NULL;
            m_G = NULL;
            m_GT = NULL;
        }
        if (m_R1 == NULL) {
            m_R1 = m_module->newMatrix(A);
            m_Q1 = m_module->newMatrix(A);
            m_G = m_module->newMatrix(A);
            m_GT = m_module->newMatrix(A);
        }

        tR1 = m_R1;
        tQ1 = m_Q1;

        host::CopyMatrix(m_R1, A);
        m_module->getMatrixUtils()->setIdentityMatrix(m_Q1);
        for (uintt fa = 0; fa < A->columns - 1; ++fa) {
            for (uintt fb = A->rows - 1; fb > fa; --fb) {
                floatt rev = m_R1->reValues[fa + fb * m_R1->columns];
                floatt imv = 0;
                if (m_R1->imValues) {
                    imv = m_R1->imValues[fa + fb * m_R1->columns];
                }
                if ((fabs(rev) < MATH_VALUE_LIMIT &&
                        fabs(imv) < MATH_VALUE_LIMIT)
                        == false) {
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
            host::CopyMatrix(this->m_output1, m_Q1);
        }
        if (this->m_output2 != m_R1) {
            host::CopyMatrix(this->m_output2, m_R1);
        }
        m_R1 = tR1;
        m_Q1 = tQ1;
    }

    QRDecompositionCpu::QRDecompositionCpu() :
    IQRDecomposition(HostMatrixModules::GetInstance()),
    m_R1(NULL), m_Q1(NULL), m_G(NULL), m_GT(NULL) {
    }

    QRDecompositionCpu::~QRDecompositionCpu() {
        if (m_R1) {
            m_module->getMatrixAllocator()->deleteMatrix(m_R1);
            m_module->getMatrixAllocator()->deleteMatrix(m_Q1);
            m_module->getMatrixAllocator()->deleteMatrix(m_G);
            m_module->getMatrixAllocator()->deleteMatrix(m_GT);
        }
    }
}
