/*
 * Copyright 2016, 2017 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#include "MathOperationsCpu.h"
#include "ThreadData.h"
namespace math {

void MultiplicationConstOperationCpu::execute() {
    intt threadsCount = utils::mapper::createThreadsMap(getBMap(),
        m_threadsCount,
        m_output->columns - m_subcolumns[0],
        m_output->rows - m_subrows[0]);
    ThreadData<MultiplicationConstOperationCpu>* threads =
        m_threadData;
    for (uintt fa = 0; fa < threadsCount; fa++) {
        threads[fa].outputs[0] = m_output;
        threads[fa].params[0] = m_matrix;
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
        threads[fa].calculateRanges(m_subcolumns, m_subrows, getBMap(), fa);
        threads[fa].thiz = this;
        threads[fa].thread.setFunction(
            MultiplicationConstOperationCpu::Execute, &threads[fa]);
        threads[fa].thread.run((this->m_threadsCount == 1));
    }
    for (uint fa = 0; fa < threadsCount; fa++) {
        threads[fa].thread.yield();
    }
    this->m_revalue = NULL;
    this->m_imvalue = NULL;
}

void MultiplicationConstOperationCpu::Execute(void* ptr) {
    ThreadData<MultiplicationConstOperationCpu>* threadData =
        (ThreadData<MultiplicationConstOperationCpu>*) ptr;
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
                int index = fa + fb * threadData->outputs[0]->columns;
                int index1 = fa + fb * threadData->params[0]->columns;
                threadData->outputs[0]->reValues[index] =
                    threadData->params[0]->reValues[index1] *
                    *(rep) -
                    threadData->params[0]->imValues[index1] *
                    *(imp);
                threadData->outputs[0]->imValues[index] =
                    threadData->params[0]->imValues[index1] *
                    *(rep) +
                    threadData->params[0]->reValues[index1] *
                    *(imp);
            }
        }
    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
        rep && !imp) {
        for (uint fa = begin; fa < end; fa++) {
            for (uint fb = begin1; fb < end1; fb++) {
                int index = fa + fb * threadData->outputs[0]->columns;
                int index1 = fa + fb * threadData->params[0]->columns;
                threadData->outputs[0]->reValues[index] =
                    threadData->params[0]->reValues[index1] *
                    *(rep);
                threadData->outputs[0]->imValues[index] =
                    threadData->params[0]->imValues[index1] *
                    *(rep);
            }
        }
    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        rep && !imp) {
        for (uint fa = begin; fa < end; fa++) {
            for (uint fb = begin1; fb < end1; fb++) {
                int index = fa + fb * threadData->outputs[0]->columns;
                int index1 = fa + fb * threadData->params[0]->columns;
                threadData->outputs[0]->reValues[index] =
                    threadData->params[0]->reValues[index1] *
                    *(rep);
            }
        }
    } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
        rep && !imp) {
        for (uint fa = begin; fa < end; fa++) {
            for (uint fb = begin1; fb < end1; fb++) {
                int index = fa + fb * threadData->outputs[0]->columns;
                int index1 = fa + fb * threadData->params[0]->columns;
                threadData->outputs[0]->imValues[index] =
                    threadData->params[0]->imValues[index1] * *(rep);
            }
        }
    }
}
}
