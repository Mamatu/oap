#include "MathOperations.h"
#include "HostMatrixModules.h"        
namespace math {

    Status IDotProductOperation::beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IDotProductOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = m_module->getMatrixUtils();
        MatrixCopier* matrixCopier = m_module->getMatrixCopier();
        if (isNotNull(matrix1, matrixUtils) == false ||
                isNotNull(matrix2, matrixUtils) == false) {
            if (isNotNull(output, matrixUtils) == false) {
                executionPath = EXECUTION_NOTHING;
            } else {
                executionPath = EXECUTION_OUTPUT_TO_ZEROS;
            }
        } else {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }

    Status IDotProductOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output,
                    this->m_matrix1, this->m_matrix2,
                    CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output, this->m_matrix1,
                        this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
                if (status == STATUS_OK) {
                    host::SetSubs(m_output, m_subcolumns, m_subrows);
                    host::SetSubColumns(m_matrix1, m_subcolumns);
                    host::SetSubRows(m_matrix2, m_subrows);
                }
            }
        }
        return status;
    }
}