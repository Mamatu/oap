#include "MathOperations.h"        
namespace math {

    Status ISubstracionOperation::beforeExecution() {
        Status status = TwoMatricesOperations::beforeExecution();
        if (status == STATUS_OK) {
            status = this->beforeExecution(this->m_output, this->m_matrix1,
                    this->m_matrix2, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output, this->m_matrix1,
                        this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
                if (status == STATUS_OK) {
                    m_matrixStructureUtils->setSubColumns(m_outputStructure,
                            this->m_subcolumns);
                    m_matrixStructureUtils->setSubRows(m_outputStructure,
                            this->m_subrows);
                }
            }
        }
        return status;
    }

    Status ISubstracionOperation::beforeExecution(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*copy)(math::Matrix* src, math::Matrix* dst,
            MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            ISubstracionOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = m_module->getMatrixUtils();
        MatrixCopier* matrixCopier = m_module->getMatrixCopier();
        if (isNotNull(matrix1, matrixUtils) == false && isNotNull(matrix2, matrixUtils) == false) {
            if (isNotNull(output, matrixUtils)) {
                executionPath = EXECUTION_OUTPUT_TO_ZEROS;
            } else {
                executionPath = EXECUTION_NOTHING;
            }
        } else if (isNotNull(matrix1, matrixUtils) == false && isNotNull(matrix2, matrixUtils) != false) {
            if (isNotNull(output, matrixUtils) == false) {
                status = STATUS_INVALID_PARAMS;
            } else {
                if (copy(output, matrix2, matrixCopier, this) != true) {
                    status = STATUS_INVALID_PARAMS;
                } else {
                    executionPath = EXECUTION_MULTIPLY_BY_MINUS_ONE;
                }
            }
        } else if (isNotNull(matrix1, matrixUtils) != false && isNotNull(matrix2, matrixUtils) == false) {
            if (isNotNull(output, matrixUtils) == false) {
                status = STATUS_INVALID_PARAMS;
            } else {
                if (copy(output, matrix1, matrixCopier, this) != true) {
                    status = STATUS_INVALID_PARAMS;
                } else {
                    executionPath = EXECUTION_NOTHING;
                }
            }
        } else if (isNotNull(matrix1, matrixUtils) && isNotNull(matrix2, matrixUtils)) {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}