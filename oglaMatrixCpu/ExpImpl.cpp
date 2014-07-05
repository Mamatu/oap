#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {
    namespace cpu {

        void ExpOperation::execute() {
            fprintf(stderr,"%s %s %d \n",__FUNCTION__,__FILE__,__LINE__);
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
            HostMatrixPrinter printer;
            printer.printReMatrix(matrix1);
            serieLimit = 100;
            fprintf(stderr,"%s %s %d \n",__FUNCTION__,__FILE__,__LINE__);
            for (uint fa = 2; fa<this->serieLimit; fa++) {
fprintf(stderr,"%s %s %d \n",__FUNCTION__,__FILE__,__LINE__);
                math::Matrix* mo = fa % 2 == 0 ? m3 : m2;
                math::Matrix* mp = fa % 2 == 0 ? m2 : m3;
                this->multiplicationOperation.setOutputMatrix(mo);
                this->multiplicationOperation.setMatrix1(mp);
                this->multiplicationOperation.setMatrix2(m1);
                this->multiplicationOperation.start();
                this->multiplicationConstOperation.setOutputMatrix(m4);
                this->multiplicationConstOperation.setMatrix(mo);
                floatt temp = 1. / factorial;
                this->multiplicationConstOperation.setReValue(&temp);
                this->multiplicationConstOperation.start();
                factorial = factorial * (fa);
                this->additionOperation.setOutputMatrix(this->m_output);
                this->additionOperation.setMatrix1(this->m_output);
                this->additionOperation.setMatrix2(m4);
                this->additionOperation.start();
                //matrixHostMem.PrintHostMatrix(this->output);
            }
            matrixAllocator.deleteMatrix(matrix2);
            matrixAllocator.deleteMatrix(matrix3);
            matrixAllocator.deleteMatrix(matrix4);
        }
    }
}