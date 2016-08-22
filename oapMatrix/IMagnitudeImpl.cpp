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
namespace math {

    Status IMagnitudeOperation::beforeExecution() {
        Status status = MatrixOperationOutputValue::beforeExecution();
        if (status == STATUS_OK) {
            MatrixUtils* matrixUtils = this->m_module->getMatrixUtils();
            bool isIm = matrixUtils->isImMatrix(this->m_matrix);
            bool isRe = matrixUtils->isReMatrix(this->m_matrix);
            if (isRe) {
                this->m_executionPathRe = EXECUTION_NORMAL;
            } else {
                this->m_executionPathRe = EXECUTION_IS_ZERO;
            }
            if (isIm) {
                this->m_executionPathIm = EXECUTION_NORMAL;
            } else {
                this->m_executionPathIm = EXECUTION_IS_ZERO;
            }
        }
        return status;
    }
}
