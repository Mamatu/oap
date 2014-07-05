#include "MathOperations.h"        
namespace math {

    Status ITensorProductOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
            }
        }
        return status;
    }

    Status ITensorProductOperation::beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            ITensorProductOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = this->m_matrixModule->getMatrixUtils();
        if (isNotNull(matrix1, matrixUtils) == false || isNotNull(matrix2, matrixUtils) == false) {
            if (isNotNull(output, matrixUtils) == false) {
                executionPath = EXECUTION_NOTHING;
            } else {
                executionPath = EXECUTION_ZEROS_TO_OUTPUT;
            }
        } else {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}