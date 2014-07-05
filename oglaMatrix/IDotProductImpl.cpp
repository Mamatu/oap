#include "MathOperations.h"        
namespace math {

    Status IDotProductOperation::beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IDotProductOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = m_matrixModule->getMatrixUtils();
        MatrixCopier* matrixCopier = m_matrixModule->getMatrixCopier();
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
            status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->beforeExecution(this->m_output, this->m_matrix1, this->m_matrix2, CopyIm, IsIm, m_executionPathIm);
                if (status == STATUS_OK) {
                    m_matrixStructureUtils->setSub(this->m_outputStructure,
                            this->m_subcolumns, this->m_subrows);
                    m_matrixStructureUtils->setMatrix(m_matrixStructure1,
                            this->m_matrix1);
                    m_matrixStructureUtils->setSubColumns(m_matrixStructure1,
                            this->m_subcolumns);
                    m_matrixStructureUtils->setMatrix(m_matrixStructure2,
                            this->m_matrix2);
                    m_matrixStructureUtils->setSubRows(m_matrixStructure2,
                            this->m_subrows);
                    if (m_matrixStructureUtils->isValid(m_outputStructure) &&
                            m_matrixStructureUtils->isValid(m_matrixStructure1) &&
                            m_matrixStructureUtils->isValid(m_matrixStructure2)) {
                    } else {
                        status = STATUS_INVALID_PARAMS;
                    }
                }
            }
        }
        return status;
    }
}