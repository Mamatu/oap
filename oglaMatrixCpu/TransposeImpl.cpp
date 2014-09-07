#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {

    void TransposeOperationCpu::Execute(void* ptr) {
        ThreadData<TransposeOperationCpu>* threadData =
                (ThreadData<TransposeOperationCpu>*) ptr;
        floatt output = 0;
        intt bcolumn = threadData->begins[0];
        uintt brow = threadData->begins[1];
        uintt ecolumn = threadData->ends[0];
        uintt erow = threadData->ends[1];
        const MatrixStructure* m_outputStructure =
                threadData->thiz->m_outputStructure;
        const MatrixStructure* m_matrixStructure =
                threadData->thiz->m_matrixStructure;
        const math::Matrix* m_output = m_outputStructure->m_matrix;
        const math::Matrix* m_matrix = m_matrixStructure->m_matrix;
        uintt columns = m_matrix->columns;
        uintt columns1 = m_output->columns;
        if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (uintt fa = brow;
                    fa < erow; fa++) {
                for (uintt fb = bcolumn;
                        fb < ecolumn; fb++) {
                    floatt value = m_matrix->reValues[fb * columns + fa];
                    floatt value1 = m_matrix->imValues[fb * columns + fa];
                    m_output->reValues[fa * columns1 + fb] = value;
                    m_output->imValues[fa * columns1 + fb] = -value1;
                }
            }

        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (uintt fa = brow;
                    fa < erow; fa++) {
                for (uintt fb = bcolumn;
                        fb < ecolumn; fb++) {
                    floatt value = m_matrix->reValues[fb * columns + fa];
                    m_output->reValues[fa * columns1 + fb] = value;
                }
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (uintt fa = m_outputStructure->m_beginRow;
                    fa < m_outputStructure->m_subrows; fa++) {
                for (uintt fb = bcolumn;
                        fb < ecolumn; fb++) {
                    floatt value1 = m_matrix->imValues[fb * columns + fa];
                    m_output->imValues[fa * columns1 + fb] = -value1;
                }
            }
        }
    }

    void TransposeOperationCpu::execute() {
        uintt threadsCount = utils::mapper::createThreadsMap(getBMap(),
                this->m_threadsCount, m_outputStructure->m_subcolumns,
                m_outputStructure->m_subrows);
        ThreadData<TransposeOperationCpu>* threads = m_threadData;
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].params[0] = m_matrixStructure;
            threads[fa].calculateRanges(m_matrixStructure, getBMap(), fa);
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(TransposeOperationCpu::Execute, &threads[fa]);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uint fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.yield();
        }
    }
}
