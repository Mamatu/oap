#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {
    namespace cpu {

        DeterminantOperation::DeterminantOperation() :
        math::IDeterminantOperation(&(HostMatrixModules::GetInstance()),
        HostMatrixStructureUtils::GetInstance(&(HostMatrixModules::GetInstance()))),
        m_q(NULL), m_r(NULL) {

        }

        DeterminantOperation::~DeterminantOperation() {
            if (m_q) {
                m_matrixModule->getMatrixAllocator()->deleteMatrix(m_q);
                m_matrixModule->getMatrixAllocator()->deleteMatrix(m_r);
            }
        }

        math::Status DeterminantOperation::beforeExecution() {
            math::Status status = IDeterminantOperation::beforeExecution();
            if (status == math::STATUS_OK) {
                if (m_matrixStructure->m_matrix->rows !=
                        m_matrixStructure->m_matrix->columns) {
                    status = math::STATUS_NOT_SUPPORTED_SUBMATRIX;
                } else {
                    if (m_q != NULL && (m_q->rows != m_matrixStructure->m_matrix->rows ||
                            m_q->columns != m_matrixStructure->m_matrix->columns)) {
                        m_matrixModule->getMatrixAllocator()->deleteMatrix(m_q);
                        m_matrixModule->getMatrixAllocator()->deleteMatrix(m_r);
                        m_q = NULL;
                        m_r = NULL;
                    }
                    if (m_q == NULL) {
                        m_q = m_matrixModule->newMatrix(m_matrixStructure->m_matrix);
                        m_r = m_matrixModule->newMatrix(m_matrixStructure->m_matrix);
                    }
                }
            }
            return status;
        }

        void DeterminantOperation::execute() {
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
}