#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {

    void DotProductOperationCpu::execute() {
        uintt threadsCount = utils::mapper::createThreadsMap(getBMap(),
                this->m_threadsCount,
                m_outputStructure->m_subcolumns,
                m_outputStructure->m_subrows);
        ThreadData<DotProductOperationCpu>* threads = m_threadData;
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_outputStructure;
            threads[fa].params[0] = m_matrixStructure1;
            threads[fa].params[1] = m_matrixStructure2;
            threads[fa].calculateRanges(m_outputStructure, getBMap(), fa);
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(DotProductOperationCpu::Execute, &threads[fa]);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.yield();
        }
    }

    void DotProductOperationCpu::Execute(void* ptr) {
        ThreadData<DotProductOperationCpu>* threadData =
                (ThreadData<DotProductOperationCpu>*) ptr;
        intt bcolumn = threadData->begins[0];
        uintt brow = threadData->begins[1];
        uintt ecolumn = threadData->ends[0];
        uintt erow = threadData->ends[1];
        const uintt columns1 = threadData->params[0]->m_matrix->columns;
        const uintt columns2 = threadData->params[1]->m_matrix->columns;
        const uintt outputColumns = threadData->outputs[0]->m_matrix->columns;
        uintt offset = columns1;
        if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
                for (uintt fb = brow; fb < erow; ++fb) {
                    floatt retemp = 0;
                    floatt imtemp = 0;
                    for (uintt fa1 = 0; fa1 < offset; ++fa1) {
                        uintt index = fa1 + columns1 * fb;
                        uintt index1 = fa1 * columns2 + fa;
                        retemp += threadData->params[0]->m_matrix->reValues[index] *
                                threadData->params[1]->m_matrix->reValues[index1];
                        retemp -= threadData->params[0]->m_matrix->imValues[index] *
                                threadData->params[1]->m_matrix->imValues[index1];
                        imtemp += threadData->params[0]->m_matrix->reValues[index] *
                                threadData->params[1]->m_matrix->imValues[index1];
                        imtemp += threadData->params[0]->m_matrix->imValues[index] *
                                threadData->params[1]->m_matrix->reValues[index1];
                    }
                    threadData->outputs[0]->m_matrix->reValues[fa + outputColumns * fb] = retemp;
                    threadData->outputs[0]->m_matrix->imValues[fa + outputColumns * fb] = imtemp;
                }
            }
        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
                for (uintt fb = brow; fb < erow; ++fb) {
                    floatt retemp = 0;
                    for (uintt fa1 = 0; fa1 < offset; ++fa1) {
                        uintt index = fa1 + columns1 * fb;
                        uintt index1 = fa + columns2 *fa1;
                        retemp += threadData->params[0]->m_matrix->reValues[index] *
                                threadData->params[1]->m_matrix->reValues[index1];
                    }
                    threadData->outputs[0]->m_matrix->reValues[fa + outputColumns * fb] = retemp;
                }
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
                for (uintt fb = brow; fb < erow; ++fb) {
                    floatt retemp = 0;
                    for (uintt fa1 = 0; fa1 < offset; ++fa1) {
                        retemp +=
                                -threadData->params[0]->m_matrix->imValues[fa1 + columns1 * fb] *
                                threadData->params[1]->m_matrix->imValues[fa1 * columns2 + fa];
                    }
                    threadData->outputs[0]->m_matrix->reValues[fa + outputColumns * fb] = retemp;
                }
            }
        }
    }
}
