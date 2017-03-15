/*
 * Copyright 2016 Marcin Matula
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
#include "HostMatrixUtils.h"        
namespace math {

    Status IAdditionOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output,
                    this->m_matrix1,
                    this->m_matrix2,
                    IsRe,
                    m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output,
                        this->m_matrix1,
                        this->m_matrix2,
                        IsIm,
                        m_executionPathIm);
                if (status == STATUS_OK) {
                    host::SetSubs(m_output, m_subcolumns[1], m_subrows[1]);
                }
            }
        }
        return status;
    }

    Status IAdditionOperation::beforeExecution(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*HasInstance)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IAdditionOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = m_module->getMatrixUtils();
        MatrixCopier* matrixCopier = m_module->getMatrixCopier();
        if (HasInstance(matrix1, matrixUtils) == false &&
                HasInstance(matrix2, matrixUtils) == false) {
            if (HasInstance(output, matrixUtils) != false) {
                executionPath = EXECUTION_NOTHING;
            }
        } else if (HasInstance(matrix1, matrixUtils) == false &&
                HasInstance(matrix2, matrixUtils)) {
            if (HasInstance(output, matrixUtils) == false) {
                status = STATUS_INVALID_PARAMS;
            } else {
                executionPath = EXECUTION_COPY_SECOND_PARAM;
            }
        } else if (HasInstance(matrix1, matrixUtils) &&
                HasInstance(matrix2, matrixUtils) == false) {
            if (HasInstance(output, matrixUtils) == false) {
                status = STATUS_INVALID_PARAMS;
            } else {
                executionPath = EXECUTION_COPY_FIRST_PARAM;
            }
        } else if (HasInstance(matrix1, matrixUtils) &&
                HasInstance(matrix2, matrixUtils)) {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}
