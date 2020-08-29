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



#include "MathOperations.h"
#include "oapHostMatrixUtils.h"        
namespace math {

Status IDotProductOperation::beforeExecution(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2,
    bool(*copy)(math::Matrix* src, math::Matrix* dst, math::IMathOperation* thiz),
    bool(*isNotNull)(math::Matrix* matrix),
    IDotProductOperation::ExecutionPath& executionPath) {
    Status status = STATUS_OK;
    //MatrixUtils* matrixUtils = m_module->getMatrixUtils();
    //MatrixCopier* matrixCopier = m_module->getMatrixCopier();
    if (isNotNull(matrix1) == false || isNotNull(matrix2) == false) {
        if (isNotNull(output) == false) {
            executionPath = EXECUTION_NOTHING;
        } else {
            executionPath = EXECUTION_OUTPUT_TO_ZEROS;
        }
    } else {
        executionPath = EXECUTION_NORMAL;
    }
    return status;
}

Status IDotProductOperation::beforeExecution() {
    Status status = TwoMatricesOperations::beforeExecution();
    if (status == STATUS_OK) {
        status = this->beforeExecution(this->m_output,
            this->m_matrix1, this->m_matrix2,
            CopyRe, IsRe, m_executionPathRe);
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output, this->m_matrix1,
                this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
            if (status == STATUS_OK) {
                oap::host::SetSubs(m_output, m_subcolumns[1], m_subrows[1]);
                oap::host::SetSubRows(m_matrix1, m_subrows[1]);
                oap::host::SetSubColumns(m_matrix2, m_subcolumns[1]);
                if (m_offset[1] == MATH_UNDEFINED) {
                    if (gColumns (m_matrix1) == gRows (m_matrix2)) {
                        m_offset[1] = gColumns (m_matrix1);
                    } else {
                        debugAssert(false);
                    }
                } else {
                    if (m_offset[1] > gColumns (m_matrix1)
                        || m_offset[1] > gRows (m_matrix2)) {
                        debugAssert(false);
                    }
                }
            }
        }
    }
    return status;
}

Status IDotProductOperation::afterExecution() {
    m_offset[0] = 0;
    m_offset[1] = MATH_UNDEFINED;
    return TwoMatricesOperations::afterExecution();
}

void IDotProductOperation::setOffset(uintt offset) {
    m_offset[1] = offset;
}

void IDotProductOperation::setOffset(uintt boffset, uintt eoffset) {
    m_offset[0] = boffset;
    m_offset[1] = eoffset;
}
}
