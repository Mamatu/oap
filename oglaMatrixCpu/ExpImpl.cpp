#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {
    namespace cpu {

        void ExpOperation::execute() {
            math::Matrix* matrix1 = this->m_matrix;
            MatrixAllocator& matrixAllocator = *m_matrixModule->getMatrixAllocator();
            MatrixCopier& matrixCopier = *m_matrixModule->getMatrixCopier();
            math::Matrix* matrix2 = host::NewMatrixCopy(matrix1);
            math::Matrix* matrix3 = host::NewMatrixCopy(matrix1);
            math::Matrix* matrix4 = host::NewMatrixCopy(matrix1);
            floatt factorial = 1;
            math::Matrix* m1 = matrix1;
            math::Matrix* m2 = matrix2;
            math::Matrix* m3 = matrix3;
            math::Matrix* m4 = matrix4;
            matrixCopier.copyMatrixToMatrix(this->m_output, matrix1);
            serieLimit = 10;
            for (uintt fa = 2; fa<this->serieLimit; ++fa) {
                math::Matrix* mo = fa % 2 == 0 ? m3 : m2;
                math::Matrix* mp = fa % 2 == 0 ? m2 : m3;
                this->dotProduct.setOutputMatrix(mo);
                this->dotProduct.setMatrix1(mp);
                this->dotProduct.setMatrix2(m1);
                this->dotProduct.start();
                this->multiplication.setOutputMatrix(m4);
                this->multiplication.setMatrix(mo);
                floatt temp = 1. / factorial;
                this->multiplication.setReValue(&temp);
                this->multiplication.start();
                factorial = factorial * (fa);
                this->addition.setOutputMatrix(this->m_output);
                this->addition.setMatrix1(this->m_output);
                this->addition.setMatrix2(m4);
                this->addition.start();
                //matrixHostMem.PrintHostMatrix(this->output);
            }
            matrixAllocator.deleteMatrix(matrix2);
            matrixAllocator.deleteMatrix(matrix3);
            matrixAllocator.deleteMatrix(matrix4);
        }
    }
}