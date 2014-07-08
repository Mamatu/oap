#include "MathOperationsCpu.h"
#include "HostMatrixStructure.h"
#include "Internal.h"
#include <math.h>
namespace math {
    namespace cpu {

        inline void switchPointer(math::Matrix*& a, math::Matrix*& b) {
            math::Matrix* temp = b;
            b = a;
            a = temp;
        }

        inline void QRDecomposition::prepareGMatrix(math::Matrix* A,
                uintt column, uintt row,
                math::Matrix* G) {
            m_matrixModule->getMatrixUtils()->setIdentityMatrix(G);
            m_matrixModule->getMatrixUtils()->setIdentityMatrix(GT);
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

        void QRDecomposition::execute() {
            dotProduct.setThreadsCount(this->m_threadsCount);
            transpose.setThreadsCount(this->m_threadsCount);
            math::Matrix* A = this->m_matrixStructure->m_matrix;
            math::Matrix* Q = this->m_outputStructure1->m_matrix;
            math::Matrix* R = this->m_outputStructure2->m_matrix;

            math::Matrix* tR1 = NULL;
            math::Matrix* tQ1 = NULL;

            if (R1 != NULL && R1->rows != R->rows &&
                    R1->columns != R->columns) {
                m_matrixModule->getMatrixAllocator()->deleteMatrix(R1);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(Q1);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(G);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(GT);
                R1 = NULL;
                Q1 = NULL;
                G = NULL;
                GT = NULL;
            }
            if (R1 == NULL) {
                R1 = m_matrixModule->newMatrix(A);
                Q1 = m_matrixModule->newMatrix(A);
                G = m_matrixModule->newMatrix(A);
                GT = m_matrixModule->newMatrix(A);
            }

            tR1 = R1;
            tQ1 = Q1;

            host::CopyMatrix(R1, A);
            m_matrixModule->getMatrixUtils()->setIdentityMatrix(Q1);
            for (uintt fa = 0; fa < A->columns - 1; ++fa) {
                for (uintt fb = A->rows - 1; fb > fa; --fb) {
                    floatt rev = R1->reValues[fa + fb * R1->columns];
                    floatt imv = 0;
                    if (R1->imValues) {
                        imv = R1->imValues[fa + fb * R1->columns];
                    }
                    if ((fabs(rev) < MATH_VALUE_LIMIT &&
                            fabs(imv) < MATH_VALUE_LIMIT)
                            == false) {
                        prepareGMatrix(R1, fa, fb, G);

                        dotProduct.setMatrix2(R1);
                        dotProduct.setMatrix1(G);
                        dotProduct.setOutputMatrix(R);
                        dotProduct.start();

                        transpose.setMatrix(G);
                        transpose.setOutputMatrix(GT);
                        transpose.start();

                        dotProduct.setMatrix1(Q1);
                        dotProduct.setMatrix2(GT);
                        dotProduct.setOutputMatrix(Q);
                        dotProduct.start();

                        switchPointer(R1, R);
                        switchPointer(Q1, Q);
                    }
                }
            }
            if (this->m_outputStructure1->m_matrix != Q1) {
                host::CopyMatrix(this->m_outputStructure1->m_matrix, Q1);
            }
            if (this->m_outputStructure2->m_matrix != R1) {
                host::CopyMatrix(this->m_outputStructure2->m_matrix, R1);
            }
            R1 = tR1;
            Q1 = tQ1;
        }

        QRDecomposition::QRDecomposition() :
        IQRDecomposition(&(HostMatrixModules::GetInstance()),
        HostMatrixStructureUtils::GetInstance(&(HostMatrixModules::GetInstance()))),
        R1(NULL), Q1(NULL), G(NULL), GT(NULL) {
        }

        QRDecomposition::~QRDecomposition() {
            if (R1) {
                m_matrixModule->getMatrixAllocator()->deleteMatrix(R1);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(Q1);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(G);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(GT);
            }
        }
    }
}
