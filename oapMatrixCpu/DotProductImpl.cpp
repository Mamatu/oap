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

void DotProductOperationCpu::execute() {
    uintt threadsCount = oap::utils::mapper::createThreadsMap(getBMap(),
        this->m_threadsCount,
        gColumns (m_output) - m_subcolumns[0],
        gRows (m_output) - m_subrows[0]);
    ThreadData<DotProductOperationCpu>* threads = m_threadData;
    for (uintt fa = 0; fa < threadsCount; fa++) {
        threads[fa].outputs[0] = m_output;
        threads[fa].params[0] = m_matrix1;
        threads[fa].params[1] = m_matrix2;
        threads[fa].calculateRanges(m_subcolumns, m_subrows, getBMap(), fa);
        threads[fa].thiz = this;
        threads[fa].thread.setFunction(DotProductOperationCpu::Execute,
            &threads[fa]);
        threads[fa].thread.run((this->m_threadsCount == 1));
    }
    for (uintt fa = 0; fa < threadsCount; fa++) {
        threads[fa].thread.join();
    }
}

void DotProductOperationCpu::Execute(void* ptr) {
    ThreadData<DotProductOperationCpu>* threadData =
        (ThreadData<DotProductOperationCpu>*) ptr;
    intt bcolumn = threadData->begins[0];
    uintt brow = threadData->begins[1];
    uintt ecolumn = threadData->ends[0];
    uintt erow = threadData->ends[1];
    const uintt realColumns1 = gMemoryColumns (threadData->params[0].m_matrix);
    const uintt realColumns2 = gMemoryColumns (threadData->params[1].m_matrix);
    const uintt outputColumns = gMemoryColumns (threadData->outputs[0].m_matrix);
    uintt* offset = threadData->thiz->m_offset;
    if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
            for (uintt fb = brow; fb < erow; ++fb) {
                floatt retemp = 0;
                floatt imtemp = 0;
                for (uintt fa1 = offset[0]; fa1 < offset[1]; ++fa1) {
                    retemp += GetRe (threadData->params[0].m_matrix, fa1, fb) *
                        GetRe (threadData->params[1].m_matrix, fa, fa1);
                    retemp -= GetIm (threadData->params[0].m_matrix, fa1, fb) *
                        GetIm (threadData->params[1].m_matrix, fa, fa1);
                    imtemp += GetRe (threadData->params[0].m_matrix, fa1, fb) *
                        GetIm (threadData->params[1].m_matrix, fa, fa1);
                    imtemp += GetIm (threadData->params[0].m_matrix, fa1, fb) *
                        GetRe (threadData->params[1].m_matrix, fa, fa1);
                }
                uintt index = fa + outputColumns * fb;
                *GetRePtr (threadData->outputs[0].m_matrix, fa, fb) = retemp;
                *GetImPtr (threadData->outputs[0].m_matrix, fa, fb) = imtemp;
            }
        }
    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
        for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
            for (uintt fb = brow; fb < erow; ++fb) {
                floatt retemp = 0;
                for (uintt fa1 = offset[0]; fa1 < offset[1]; ++fa1) {
                    retemp += GetRe (threadData->params[0].m_matrix, fa1, fb) * GetRe (threadData->params[1].m_matrix, fa, fa1);
                }
                *GetRePtr (threadData->outputs[0].m_matrix, fa, fb) = retemp;
            }
        }
    } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
            for (uintt fb = brow; fb < erow; ++fb) {
                floatt retemp = 0;
                for (uintt fa1 = offset[0]; fa1 < offset[1]; ++fa1) {
                    uintt index = fa1 + realColumns1 * fb;
                    uintt index1 = fa1 * realColumns2 + fa;
                    retemp += -GetIm (threadData->params[0].m_matrix, fa1, fb) * GetIm (threadData->params[1].m_matrix, fa, fa1);
                }
                *GetRePtr (threadData->outputs[0].m_matrix, fa, fb) = retemp;
            }
        }
    }
}
}
