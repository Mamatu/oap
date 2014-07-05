#include "MathOperations.h"

namespace math {

    Status IDeterminantOperation::prepare(floatt* output, math::Matrix* matrix,
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            ExecutionPath& executionPath) {
        if (isNotNull(matrix, m_matrixModule->getMatrixUtils()) == true) {
            executionPath = EXECUTION_NORMAL;
        } else {
            executionPath = EXECUTION_NOTHING;
        }
        if (m_matrixModule->getMatrixUtils()->getColumns(matrix) !=
                m_matrixModule->getMatrixUtils()->getRows(matrix)) {
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
