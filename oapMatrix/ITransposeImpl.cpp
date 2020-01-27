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
#include "oapHostMatrixUtils.h"        
namespace math {

    bool isMatrix(math::Matrix* m) { return m != NULL && gReValues (m) != NULL && gImValues (m) != NULL; }
    bool isReMatrix(math::Matrix* m) { return m != NULL && gReValues (m) != NULL && gImValues (m) == NULL; }
    bool isImMatrix(math::Matrix* m) {return m != NULL && gReValues (m) == NULL && gImValues (m) != NULL; }

    Status ITransposeOperation::beforeExecution() {
        Status status = MatrixOperationOutputMatrix::beforeExecution();
        if (status == STATUS_OK) {
            oap::host::SetSubs(m_output, m_subcolumns[1], m_subrows[1]);
            if (m_output != m_matrix) {
                if (isMatrix(m_output) && isMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NORMAL;
                } else if (isReMatrix(m_output) && isReMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NOTHING;
                } else if (isImMatrix(m_output) && isImMatrix(m_matrix)) {
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
