#include "MathOperations.h"        
namespace math {

    Status IDiagonalizationOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
            }
        }
        return status;
    }

    Status IDiagonalizationOperation::beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IDiagonalizationOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = this->m_matrixModule->getMatrixUtils();
        MatrixCopier* matrixCopier = this->m_matrixModule->getMatrixCopier();
        if (isNotNull(matrix1, matrixUtils) == false && isNotNull(matrix1, matrixUtils) == false) {
            status = STATUS_INVALID_PARAMS;
        } else {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}