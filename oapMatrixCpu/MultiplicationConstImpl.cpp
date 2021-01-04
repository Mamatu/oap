/*
 * Copyright 2016 - 2021 Marcin Matula
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
#include "MatrixAPI.h"

namespace math {

void MultiplicationConstOperationCpu::execute() {
    intt threadsCount = oap::utils::mapper::createThreadsMap(getBMap(),
        m_threadsCount,
        gColumns (m_output) - m_subcolumns[0],
        gRows (m_output) - m_subrows[0]);
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
        threads[fa].thread.join();
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
                int index = fa + fb * gColumns (threadData->outputs[0].m_matrix);
                int index1 = fa + fb * gColumns (threadData->params[0].m_matrix);
                *GetRePtr (threadData->outputs[0].m_matrix, fa, fb) =
                    GetRe (threadData->params[0].m_matrix, fa, fb) *
                    *(rep) -
                    GetIm (threadData->params[0].m_matrix, fa, fb) *
                    *(imp);
                *GetImPtr (threadData->outputs[0].m_matrix, fa, fb) =
                    GetIm (threadData->params[0].m_matrix, fa, fb) *
                    *(rep) +
                    GetRe (threadData->params[0].m_matrix, fa, fb) *
                    *(imp);
            }
        }
    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
        rep && !imp) {
        for (uint fa = begin; fa < end; fa++) {
            for (uint fb = begin1; fb < end1; fb++) {
                int index = fa + fb * gColumns (threadData->outputs[0].m_matrix);
                int index1 = fa + fb * gColumns (threadData->params[0].m_matrix);
                *GetRePtrIndex (threadData->outputs[0].m_matrix, index) =
                    GetReIndex (threadData->params[0].m_matrix, index1) *
                    *(rep);
                *GetImPtrIndex (threadData->outputs[0].m_matrix, index) =
                    GetImIndex (threadData->params[0].m_matrix, index1) *
                    *(rep);
            }
        }
    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        rep && !imp) {
        for (uint fa = begin; fa < end; fa++) {
            for (uint fb = begin1; fb < end1; fb++) {
                int index = fa + fb * gColumns (threadData->outputs[0].m_matrix);
                int index1 = fa + fb * gColumns (threadData->params[0].m_matrix);
                *GetRePtrIndex (threadData->outputs[0].m_matrix, index) =
                    GetReIndex (threadData->params[0].m_matrix, index1) *
                    *(rep);
            }
        }
    } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL &&
        rep && !imp) {
        for (uint fa = begin; fa < end; fa++) {
            for (uint fb = begin1; fb < end1; fb++) {
                int index = fa + fb * gColumns (threadData->outputs[0].m_matrix);
                int index1 = fa + fb * gColumns (threadData->params[0].m_matrix);
                *GetImPtrIndex (threadData->outputs[0].m_matrix, index) =
                    GetImIndex (threadData->params[0].m_matrix, index1) * *(rep);
            }
        }
    }
}
}
