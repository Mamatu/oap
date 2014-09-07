#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {

    DeterminantOperationCpu::DeterminantOperationCpu() :
    math::IDeterminantOperation(
    HostMatrixModules::GetInstance(),
    HostMatrixStructureUtils::GetInstance()),
    m_q(NULL), m_r(NULL) {

    }

    DeterminantOperationCpu::~DeterminantOperationCpu() {
        if (m_q) {
            m_module->getMatrixAllocator()->deleteMatrix(m_q);
            m_module->getMatrixAllocator()->deleteMatrix(m_r);
        }
    }

    math::Status DeterminantOperationCpu::beforeExecution() {
        math::Status status = IDeterminantOperation::beforeExecution();
        if (status == math::STATUS_OK) {
            if (m_matrixStructure->m_matrix->rows !=
                    m_matrixStructure->m_matrix->columns) {
                status = math::STATUS_NOT_SUPPORTED_SUBMATRIX;
            } else {
                if (m_q != NULL && (m_q->rows != m_matrixStructure->m_matrix->rows ||
                        m_q->columns != m_matrixStructure->m_matrix->columns)) {
                    m_module->getMatrixAllocator()->deleteMatrix(m_q);
                    m_module->getMatrixAllocator()->deleteMatrix(m_r);
                    m_q = NULL;
                    m_r = NULL;
                }
                if (m_q == NULL) {
                    m_q = m_module->newMatrix(m_matrixStructure->m_matrix);
                    m_r = m_module->newMatrix(m_matrixStructure->m_matrix);
                }
            }
        }
        return status;
    }

    void DeterminantOperationCpu::execute() {
        m_qrDecomposition.setThreadsCount(m_threadsCount);
        m_qrDecomposition.setOutputMatrix1(m_q);
        m_qrDecomposition.setOutputMatrix2(m_r);
        m_qrDecomposition.setMatrix(m_matrix);
        m_qrDecomposition.start();
        floatt det = host::GetTrace(m_r);
        if (m_output1) {
            *m_output1 = det;
        }
        if (m_output2) {
            *m_output2 = 0;
        }
    }
}