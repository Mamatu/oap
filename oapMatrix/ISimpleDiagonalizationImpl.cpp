/*
 * Copyright 2016 - 2019 Marcin Matula
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



#include "MathOperations.h"        
namespace math {

    Status IDiagonalizationOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
            }
        }
        return status;
    }

    Status IDiagonalizationOperation::beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix),
            IDiagonalizationOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        //MatrixUtils* matrixUtils = this->m_module->getMatrixUtils();
        //MatrixCopier* matrixCopier = this->m_module->getMatrixCopier();
        if (isNotNull(matrix1) == false && isNotNull(matrix1) == false) {
            status = STATUS_INVALID_PARAMS;
        } else {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}
