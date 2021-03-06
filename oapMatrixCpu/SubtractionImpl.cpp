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
namespace math {

    void SubstracionOperationCpu::execute() {
        uintt threadsCount = oap::utils::mapper::createThreadsMap(getBMap(),
                this->m_threadsCount, gColumns (m_output), gRows (m_output));
        ThreadData<SubstracionOperationCpu>* threads = m_threadData;
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_output;
            threads[fa].params[0] = m_matrix1;
            threads[fa].params[1] = m_matrix2;
            threads[fa].thiz = this;
            threads[fa].calculateRanges(m_subcolumns, m_subrows, getBMap(), fa);
            threads[fa].thread.run (SubstracionOperationCpu::Execute, &threads[fa]);
        }
        for (uint fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.stop();
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
                    *GetRePtrIndex (threadData->params[0].m_matrix, fa) =
                            GetReIndex (threadData->params[0].m_matrix, fa) -
                            GetReIndex (threadData->params[1].m_matrix, fa);
                    *GetImPtrIndex (threadData->params[0].m_matrix, fa) =
                            GetImIndex (threadData->params[0].m_matrix, fa) -
                            GetImIndex (threadData->params[1].m_matrix, fa);
                }
            }
        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (intt fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    *GetRePtrIndex (threadData->params[0].m_matrix, index) =
                            GetReIndex (threadData->params[0].m_matrix, index) -
                            GetReIndex (threadData->params[1].m_matrix, index);
                }
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    *GetImPtrIndex (threadData->params[0].m_matrix, index) =
                            GetImIndex (threadData->params[0].m_matrix, index) -
                            GetImIndex (threadData->params[1].m_matrix, index);
                }
            }
        }
        if (threadData->thiz->m_executionPathRe == EXECUTION_MULTIPLY_BY_MINUS_ONE) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    *GetRePtrIndex (threadData->params[0].m_matrix, index) =
                            -GetReIndex (threadData->params[0].m_matrix, index);
                }
            }
        }
        if (threadData->thiz->m_executionPathIm == EXECUTION_MULTIPLY_BY_MINUS_ONE) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    *GetImPtrIndex (threadData->params[0].m_matrix, index) =
                            -GetImIndex (threadData->params[0].m_matrix, index);
                }
            }
        }
        if (threadData->thiz->m_executionPathRe == EXECUTION_OUTPUT_TO_ZEROS) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    *GetRePtrIndex (threadData->params[0].m_matrix, index) = 0;
                }
            }
        }
        if (threadData->thiz->m_executionPathIm == EXECUTION_OUTPUT_TO_ZEROS) {
            for (uint fa = begin; fa < end; fa++) {
                for (intt fb = begin1; fb < end1; fb++) {
                    intt index = fa + offset * fb;
                    *GetImPtrIndex (threadData->params[0].m_matrix, index) = 0;
                }
            }
        }
    }
}
