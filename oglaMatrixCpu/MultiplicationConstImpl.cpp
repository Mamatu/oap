#include "MathOperationsCpu.h"
#include "HostMatrixStructure.h"
#include "Internal.h"
namespace math {
    namespace cpu {

        void MultiplicationConstOperation::execute() {
            intt threadsCount = utils::mapper::createThreadsMap(getBMap(),
                    m_threadsCount, m_outputStructure->m_subcolumns,
                    m_outputStructure->m_subrows);
            ThreadData<MultiplicationConstOperation>* threads =
                    m_threadData;
            for (uintt fa = 0; fa < threadsCount; fa++) {
                threads[fa].outputs[0] = m_outputStructure;
                threads[fa].params[0] = m_matrixStructure;
                if (this->m_revalue != NULL && *m_revalue != 0) {
                    threads[fa].valuesPtr[0] = this->m_revalue;
                } else {
                    threads[fa].valuesPtr[0] = NULL;
                }
                if (this->m_imvalue != NULL && *m_imvalue != 0) {
                    threads[fa].valuesPtr[1] = this->m_imvalue;
                } else {
                    threads[fa].valuesPtr[1] = NULL;
                }
                threads[fa].calculateRanges(m_outputStructure, getBMap(), fa);
                threads[fa].thiz = this;
                threads[fa].thread.setFunction(MultiplicationConstOperation::Execute, &threads[fa]);
                threads[fa].thread.run((this->m_threadsCount == 1));
            }
            for (uint fa = 0; fa < threadsCount; fa++) {
                threads[fa].thread.yield();
            }
            this->m_revalue = NULL;
            this->m_imvalue = NULL;
        }

        void MultiplicationConstOperation::Execute(void* ptr) {
            ThreadData<MultiplicationConstOperation>* threadData = (ThreadData<MultiplicationConstOperation>*) ptr;
            uint begin = threadData->begins[0];
            uint end = threadData->ends[0];
            uint begin1 = threadData->begins[1];
            uint end1 = threadData->ends[1];
            floatt* rep = threadData->valuesPtr[0];
            floatt* imp = threadData->valuesPtr[1];
            if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                    threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
                    rep && imp) {
                for (uint fa = begin; fa < end; fa++) {
                    for (uint fb = begin1; fb < end1; fb++) {
                        int index = fa + fb * threadData->outputs[0]->m_matrix->columns;
                        int index1 = fa + fb * threadData->params[0]->m_matrix->columns;
                        threadData->outputs[0]->m_matrix->reValues[index] =
                                threadData->params[0]->m_matrix->reValues[index1] *
                                *(rep) -
                                threadData->params[0]->m_matrix->imValues[index1] *
                                *(imp);
                        threadData->outputs[0]->m_matrix->imValues[index] =
                                threadData->params[0]->m_matrix->imValues[index1] *
                                *(rep) +
                                threadData->params[0]->m_matrix->reValues[index1] *
                                *(imp);
                    }
                }
            } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                    threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
                    rep && !imp) {
                for (uint fa = begin; fa < end; fa++) {
                    for (uint fb = begin1; fb < end1; fb++) {
                        int index = fa + fb * threadData->outputs[0]->m_matrix->columns;
                        int index1 = fa + fb * threadData->params[0]->m_matrix->columns;
                        threadData->outputs[0]->m_matrix->reValues[index] =
                                threadData->params[0]->m_matrix->reValues[index1] *
                                *(rep);
                        threadData->outputs[0]->m_matrix->imValues[index] =
                                threadData->params[0]->m_matrix->imValues[index1] *
                                *(rep);
                    }
                }
            } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                    rep && !imp) {
                for (uint fa = begin; fa < end; fa++) {
                    for (uint fb = begin1; fb < end1; fb++) {
                        int index = fa + fb * threadData->outputs[0]->m_matrix->columns;
                        int index1 = fa + fb * threadData->params[0]->m_matrix->columns;
                        threadData->outputs[0]->m_matrix->reValues[index] =
                                threadData->params[0]->m_matrix->reValues[index1] *
                                *(rep);
                    }
                }
            } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
                    rep && !imp) {
                for (uint fa = begin; fa < end; fa++) {
                    for (uint fb = begin1; fb < end1; fb++) {

                        int index = fa + fb * threadData->outputs[0]->m_matrix->columns;
                        int index1 = fa + fb * threadData->params[0]->m_matrix->columns;
                        threadData->outputs[0]->m_matrix->imValues[index] =
                                threadData->params[0]->m_matrix->imValues[index1] * *(rep);
                    }
                }
            }
        }
    }
}