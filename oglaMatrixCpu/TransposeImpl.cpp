#include "MathOperationsCpu.h"
#include "ThreadData.h"
namespace math {

void TransposeOperationCpu::Execute(void* ptr) {
    ThreadData<TransposeOperationCpu>* threadData =
        (ThreadData<TransposeOperationCpu>*) ptr;
    intt bcolumn = threadData->begins[0];
    uintt brow = threadData->begins[1];
    uintt ecolumn = threadData->ends[0];
    uintt erow = threadData->ends[1];
    const math::Matrix* m_output = threadData->outputs[0].m_matrix;
    const math::Matrix* m_matrix = threadData->params[0].m_matrix;
    uintt columns = m_matrix->columns;
    uintt columns1 = m_output->columns;
    if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = brow; fa < erow; fa++) {
            for (uintt fb = bcolumn; fb < ecolumn; fb++) {
                uintt index1 = fb * columns + fa;
                floatt value = m_matrix->reValues[index1];
                floatt value1 = m_matrix->imValues[index1];
                uintt indexa = fa * columns1 + fb;
                m_output->reValues[indexa] = value;
                m_output->imValues[indexa] = -value1;
            }
        }

    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
        for (uintt fa = brow; fa < erow; fa++) {
            for (uintt fb = bcolumn; fb < ecolumn; fb++) {
                floatt value = m_matrix->reValues[fb * columns + fa];
                //m_output->reValues[fa * columns1 + fb] = value;
                SetRe(m_output, fb, fa, value);
            }
        }
    } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = brow; fa < erow; fa++) {
            for (uintt fb = bcolumn; fb < ecolumn; fb++) {
                floatt value1 = m_matrix->imValues[fb * columns + fa];
                m_output->imValues[fa * columns1 + fb] = -value1;
            }
        }
    }
}

void TransposeOperationCpu::execute() {
    uintt threadsCount = utils::mapper::createThreadsMap(getBMap(),
        this->m_threadsCount, m_output->columns - m_subcolumns[0],
        m_output->rows - m_subrows[0]);
    ThreadData<TransposeOperationCpu>* threads = m_threadData;
    for (uintt fa = 0; fa < threadsCount; fa++) {
        threads[fa].outputs[0] = m_output;
        threads[fa].params[0] = m_matrix;
        threads[fa].calculateRanges(m_subcolumns, m_subrows, getBMap(), fa);
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
