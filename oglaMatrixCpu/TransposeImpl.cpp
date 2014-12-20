#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {

    void TransposeOperationCpu::Execute(void* ptr) {
        ThreadData<TransposeOperationCpu>* threadData =
                (ThreadData<TransposeOperationCpu>*) ptr;
        intt bcolumn = threadData->begins[0];
        uintt brow = threadData->begins[1];
        uintt ecolumn = threadData->ends[0];
        uintt erow = threadData->ends[1];
        const math::Matrix* m_output = threadData->outputs[0];
        const math::Matrix* m_matrix = threadData->params[0];
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
            for (uintt fa = m_output->rows;
                    fa < m_output->rows; fa++) {
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
                this->m_threadsCount, m_output->columns,
                m_output->rows);
        ThreadData<TransposeOperationCpu>* threads = m_threadData;
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_output;
            threads[fa].params[0] = m_matrix;
            threads[fa].calculateRanges(m_matrix, getBMap(), fa);
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(TransposeOperationCpu::Execute,
                    &threads[fa]);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uint fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.yield();
        }
    }
}
