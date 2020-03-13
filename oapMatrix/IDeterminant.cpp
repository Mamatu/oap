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
#include "MatrixAPI.h"

namespace math {

    Status IDeterminantOperation::prepare(floatt* output, math::Matrix* matrix,
            bool(*isNotNull)(math::Matrix* matrix),
            ExecutionPath& executionPath) {
        if (isNotNull(matrix) == true) {
            executionPath = EXECUTION_NORMAL;
        } else {
            executionPath = EXECUTION_NOTHING;
        }
        if (gColumns (matrix) != gRows (matrix)) {
            return STATUS_INVALID_PARAMS;
        }
        return STATUS_OK;
    }

    Status IDeterminantOperation::beforeExecution() {
        Status status = MatrixOperationOutputValue::beforeExecution();
        if (status == STATUS_OK) {
            status = this->prepare(this->m_output1, this->m_matrix, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->prepare(this->m_output2, this->m_matrix, IsIm, m_executionPathIm);
            }
        }
        return status;
    }
}
