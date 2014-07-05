#include "MathOperations.h"        
namespace math {

    Status IMagnitudeOperation::beforeExecution() {
        Status status = MatrixOperationOutputValue::beforeExecution();
        if (status == STATUS_OK) {
            MatrixUtils* matrixUtils = this->m_matrixModule->getMatrixUtils();
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