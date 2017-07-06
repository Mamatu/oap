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



#include "MathOperations.h"
#include "HostMatrixUtils.h"        
namespace math {

    Status ITransposeOperation::beforeExecution() {
        Status status = MatrixOperationOutputMatrix::beforeExecution();
        if (status == STATUS_OK) {
            host::SetSubs(m_output, m_subcolumns[1], m_subrows[1]);
            if (m_output != m_matrix) {
                if (m_module->getMatrixUtils()->isMatrix(m_output) &&
                        m_module->getMatrixUtils()->isMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NORMAL;
                } else if (m_module->getMatrixUtils()->isReMatrix(m_output) &&
                        m_module->getMatrixUtils()->isReMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NOTHING;
                } else if (m_module->getMatrixUtils()->isImMatrix(m_output) &&
                        m_module->getMatrixUtils()->isImMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NOTHING;
                    m_executionPathIm = EXECUTION_NORMAL;
                } else {
                    status = STATUS_INVALID_PARAMS;
                }
            } else {
                status = STATUS_INVALID_PARAMS;
            }
        }
        return status;
    }
}
