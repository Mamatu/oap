/*
 * Copyright 2016, 2017 Marcin Matula
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
namespace math {

    void ExpOperationCpu::execute() {
        math::Matrix* matrix1 = this->m_matrix;
        //MatrixAllocator& matrixAllocator = *m_module->getMatrixAllocator();
        //MatrixCopier& matrixCopier = *m_module->getMatrixCopier();
        math::Matrix* matrix2 = host::NewMatrixCopy(matrix1);
        math::Matrix* matrix3 = host::NewMatrixCopy(matrix1);
        math::Matrix* matrix4 = host::NewMatrixCopy(matrix1);
        floatt factorial = 1;
        math::Matrix* m1 = matrix1;
        math::Matrix* m2 = matrix2;
        math::Matrix* m3 = matrix3;
        math::Matrix* m4 = matrix4;
        host::CopyMatrix(this->m_output, matrix1);
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
        host::DeleteMatrix(matrix2);
        host::DeleteMatrix(matrix3);
        host::DeleteMatrix(matrix4);
    }
}
