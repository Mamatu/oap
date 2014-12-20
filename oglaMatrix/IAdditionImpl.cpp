#include "MathOperations.h"
#include "HostMatrixModules.h"        
namespace math {

    Status IAdditionOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output,
                    this->m_matrix1,
                    this->m_matrix2,
                    IsRe,
                    m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output,
                        this->m_matrix1,
                        this->m_matrix2,
                        IsIm,
                        m_executionPathIm);
                if (status == STATUS_OK) {
                    host::SetSubs(m_output, m_subcolumns, m_subrows);
                }
            }
        }
        return status;
    }

    Status IAdditionOperation::beforeExecution(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*HasInstance)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IAdditionOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = m_module->getMatrixUtils();
        MatrixCopier* matrixCopier = m_module->getMatrixCopier();
        if (HasInstance(matrix1, matrixUtils) == false &&
                HasInstance(matrix2, matrixUtils) == false) {
            if (HasInstance(output, matrixUtils) != false) {
                executionPath = EXECUTION_NOTHING;
            }
        } else if (HasInstance(matrix1, matrixUtils) == false &&
                HasInstance(matrix2, matrixUtils)) {
            if (HasInstance(output, matrixUtils) == false) {
                status = STATUS_INVALID_PARAMS;
            } else {
                executionPath = EXECUTION_COPY_SECOND_PARAM;
            }
        } else if (HasInstance(matrix1, matrixUtils) &&
                HasInstance(matrix2, matrixUtils) == false) {
            if (HasInstance(output, matrixUtils) == false) {
                status = STATUS_INVALID_PARAMS;
            } else {
                executionPath = EXECUTION_COPY_FIRST_PARAM;
            }
        } else if (HasInstance(matrix1, matrixUtils) &&
                HasInstance(matrix2, matrixUtils)) {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}
