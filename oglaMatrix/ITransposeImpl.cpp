#include "MathOperations.h"
#include "HostMatrixModules.h"        
namespace math {

    Status ITransposeOperation::beforeExecution() {
        Status status = MatrixOperationOutputMatrix::beforeExecution();
        if (status == STATUS_OK) {
            host::SetSubs(m_output, m_subcolumns[1], m_subrows[1]);
            if (m_output != m_matrix) {
                if (m_module->getMatrixUtils()->isMatrix(m_output) &&
                        m_module->getMatrixUtils()->isMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NORMAL;
                } else if (m_module->getMatrixUtils()->isReMatrix(m_output) &&
                        m_module->getMatrixUtils()->isReMatrix(m_matrix)) {
                    m_executionPathRe = EXECUTION_NORMAL;
                    m_executionPathIm = EXECUTION_NOTHING;
                } else if (m_module->getMatrixUtils()->isImMatrix(m_output) &&
                        m_module->getMatrixUtils()->isImMatrix(m_matrix)) {
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