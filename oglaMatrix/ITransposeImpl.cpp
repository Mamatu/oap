#include "MathOperations.h"        
namespace math {

    Status ITransposeOperation::beforeExecution() {
        Status status = MatrixOperationOutputMatrix::beforeExecution();
        if (status == STATUS_OK) {
            m_matrixStructureUtils->setSub(m_outputStructure,
                    m_subcolumns, m_subrows);
            if (m_output != m_matrix) {
                if (m_matrixModule->getMatrixUtils()->isMatrix(m_output) &&
                        m_matrixModule->getMatrixUtils()->isMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NORMAL;
                } else if (m_matrixModule->getMatrixUtils()->isReMatrix(m_output) &&
                        m_matrixModule->getMatrixUtils()->isReMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NOTHING;
                } else if (m_matrixModule->getMatrixUtils()->isImMatrix(m_output) &&
                        m_matrixModule->getMatrixUtils()->isImMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NOTHING;
                    m_executionPathIm = EXECUTION_NORMAL;
                } else {
                    status = STATUS_INVALID_PARAMS;
                }
            } else {
                status = STATUS_INVALID_PARAMS;
            }
        }
        return status;
    }
}