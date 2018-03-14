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

    DeterminantOperationCpu::DeterminantOperationCpu() :
    math::IDeterminantOperation(),
    m_q(NULL), m_r(NULL) {
        // not implemented
    }

    DeterminantOperationCpu::~DeterminantOperationCpu() {
        if (m_q) {
            oap::host::DeleteMatrix(m_q);
            oap::host::DeleteMatrix(m_r);
        }
    }

    math::Status DeterminantOperationCpu::beforeExecution() {
        math::Status status = IDeterminantOperation::beforeExecution();
        if (status == math::STATUS_OK) {
            if (m_matrix->rows != m_matrix->columns) {
                status = math::STATUS_NOT_SUPPORTED_SUBMATRIX;
            } else {
                if (m_q != NULL && (m_q->rows != m_matrix->rows ||
                    m_q->columns != m_matrix->columns))
                {
                    oap::host::DeleteMatrix(m_q);
                    oap::host::DeleteMatrix(m_r);
                    m_q = NULL;
                    m_r = NULL;
                }
                if (m_q == NULL) {
                    m_q = oap::host::NewMatrix(m_matrix);
                    m_r = oap::host::NewMatrix(m_matrix);
                }
            }
        }
        return status;
    }

    void DeterminantOperationCpu::execute() {
        m_qrDecomposition.setThreadsCount(m_threadsCount);
        m_qrDecomposition.setOutputMatrix1(m_q);
        m_qrDecomposition.setOutputMatrix2(m_r);
        m_qrDecomposition.setMatrix(m_matrix);
        m_qrDecomposition.start();
        floatt det = oap::host::GetTrace(m_r);
        if (m_output1) {
            *m_output1 = det;
        }
        if (m_output2) {
            *m_output2 = 0;
        }
    }
}
