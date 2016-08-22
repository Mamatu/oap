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
#include "HostMatrixModules.h"        
namespace math {

    Status IMultiplicationConstOperation::beforeExecution() {
        Status status = MatrixValueOperation::beforeExecution();
        if (status == STATUS_OK) {
            status = this->prepare(this->m_output, this->m_matrix,
                    this->m_revalue, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->prepare(this->m_output, this->m_matrix,
                        this->m_revalue, CopyIm, IsIm, m_executionPathIm);
            }
        }
        host::SetSubs(m_output, m_subcolumns[1], m_subrows[1]);
        return status;
    }

    Status IMultiplicationConstOperation::prepare(math::Matrix* output, math::Matrix* matrix1, floatt* value,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IMultiplicationConstOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = this->m_module->getMatrixUtils();
        MatrixCopier* matrixCopier = this->m_module->getMatrixCopier();
        if (isNotNull(matrix1, matrixUtils) == false || value == 0) {
            if (isNotNull(output, matrixUtils) != false) {
                executionPath = EXECUTION_ZEROS_TO_OUTPUT;
            } else {
                executionPath = EXECUTION_NOTHING;
            }
        } else {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}
