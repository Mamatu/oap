#include "MathOperations.h"        
namespace math {

    Status IMultiplicationConstOperation::beforeExecution() {
        Status status = MatrixValueOperation::beforeExecution();
        if (status == STATUS_OK) {
            status = this->prepare(this->m_output, this->m_matrix, this->m_revalue, CopyRe, IsRe, m_executionPathRe);
            if (status == STATUS_OK) {
                status = this->prepare(this->m_output, this->m_matrix, this->m_revalue, CopyIm, IsIm, m_executionPathIm);
            }
        }
        m_matrixStructureUtils->setSub(this->m_outputStructure,
                this->m_subcolumns, this->m_subrows);
        return status;
    }

    Status IMultiplicationConstOperation::prepare(math::Matrix* output, math::Matrix* matrix1, floatt* value,
            bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
            bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
            IMultiplicationConstOperation::ExecutionPath& executionPath) {
        Status status = STATUS_OK;
        MatrixUtils* matrixUtils = this->m_module->getMatrixUtils();
        MatrixCopier* matrixCopier = this->m_module->getMatrixCopier();
        if (isNotNull(matrix1, matrixUtils) == false || value == 0) {
            if (isNotNull(output, matrixUtils) != false) {
                executionPath = EXECUTION_ZEROS_TO_OUTPUT;
            } else {
                executionPath = EXECUTION_NOTHING;
            }
        } else {
            executionPath = EXECUTION_NORMAL;
        }
        return status;
    }
}