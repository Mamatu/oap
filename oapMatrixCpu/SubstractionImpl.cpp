#include "MathOperationsCpu.h"
#include "ThreadData.h"
namespace math {

    void SubstracionOperationCpu::execute() {
        uintt threadsCount = utils::mapper::createThreadsMap(getBMap(),
                this->m_threadsCount, m_output->columns, m_output->rows);
        ThreadData<SubstracionOperationCpu>* threads = m_threadData;
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_output;
            threads[fa].params[0] = m_matrix1;
            threads[fa].params[1] = m_matrix2;
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(SubstracionOperationCpu::Execute, &threads[fa]);
            threads[fa].calculateRanges(m_subcolumns, m_subrows, getBMap(), fa);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uint fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.yield();
        }
    }

    void SubstracionOperationCpu::Execute(void* ptr) {
        ThreadData<SubstracionOperationCpu>* threadData =
                (ThreadData<SubstracionOperationCpu>*) ptr;
        intt begin = threadData->begins[0];
        intt end = threadData->ends[0];
        intt begin1 = threadData->begins[1];
        intt end1 = threadData->ends[1];
        intt offset = threadData->offset;
        if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa1 = begin; fa1 < end; fa1++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt fa = fa1 + offset * fb;
                    threadData->outputs[0]->reValues[fa] =
                            threadData->params[0]->reValues[fa] -
                            threadData->params[1]->reValues[fa];
                    threadData->outputs[0]->imValues[fa] =
                            threadData->params[0]->imValues[fa] -
                            threadData->params[1]->imValues[fa];
                }
            }
        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (intt fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    threadData->outputs[0]->reValues[index] =
                            threadData->params[0]->reValues[index] -
                            threadData->params[1]->reValues[index];
                }
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    threadData->outputs[0]->imValues[index] =
                            threadData->params[0]->imValues[index] -
                            threadData->params[1]->imValues[index];
                }
            }
        }
        if (threadData->thiz->m_executionPathRe == EXECUTION_MULTIPLY_BY_MINUS_ONE) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    threadData->outputs[0]->reValues[index] =
                            -threadData->outputs[0]->reValues[index];
                }
            }
        }
        if (threadData->thiz->m_executionPathIm == EXECUTION_MULTIPLY_BY_MINUS_ONE) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    threadData->outputs[0]->imValues[index] =
                            -threadData->outputs[0]->imValues[index];
                }
            }
        }
        if (threadData->thiz->m_executionPathRe == EXECUTION_OUTPUT_TO_ZEROS) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    threadData->outputs[0]->reValues[index] = 0;
                }
            }
        }
        if (threadData->thiz->m_executionPathIm == EXECUTION_OUTPUT_TO_ZEROS) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    threadData->outputs[0]->imValues[index] = 0;
                }
            }
        }
    }
}