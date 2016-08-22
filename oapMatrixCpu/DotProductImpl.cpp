/*
 * Copyright 2016 Marcin Matula
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
    uintt threadsCount = utils::mapper::createThreadsMap(getBMap(),
        this->m_threadsCount,
        m_output->columns - m_subcolumns[0],
        m_output->rows - m_subrows[0]);
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
    const uintt realColumns1 = threadData->params[0].m_matrix->realColumns;
    const uintt realColumns2 = threadData->params[1].m_matrix->realColumns;
    const uintt outputColumns = threadData->outputs[0].m_matrix->realColumns;
    uintt* offset = threadData->thiz->m_offset;
    if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
            for (uintt fb = brow; fb < erow; ++fb) {
                floatt retemp = 0;
                floatt imtemp = 0;
                for (uintt fa1 = offset[0]; fa1 < offset[1]; ++fa1) {
                    uintt index = fa1 + realColumns1 * fb;
                    uintt index1 = fa1 * realColumns2 + fa;
                    retemp += threadData->params[0].m_matrix->reValues[index] *
                        threadData->params[1].m_matrix->reValues[index1];
                    retemp -= threadData->params[0].m_matrix->imValues[index] *
                        threadData->params[1].m_matrix->imValues[index1];
                    imtemp += threadData->params[0].m_matrix->reValues[index] *
                        threadData->params[1].m_matrix->imValues[index1];
                    imtemp += threadData->params[0].m_matrix->imValues[index] *
                        threadData->params[1].m_matrix->reValues[index1];
                }
                uintt index = fa + outputColumns * fb;
                threadData->outputs[0].m_matrix->reValues[index] = retemp;
                threadData->outputs[0].m_matrix->imValues[index] = imtemp;
            }
        }
    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
        for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
            for (uintt fb = brow; fb < erow; ++fb) {
                floatt retemp = 0;
                for (uintt fa1 = offset[0]; fa1 < offset[1]; ++fa1) {
                    uintt index = fa1 + realColumns1 * fb;
                    uintt index1 = fa1 * realColumns2 + fa;
                    retemp += threadData->params[0].m_matrix->reValues[index] *
                        threadData->params[1].m_matrix->reValues[index1];
                }
                uintt index = fa + outputColumns * fb;
                threadData->outputs[0].m_matrix->reValues[index] = retemp;
            }
        }
    } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = bcolumn; fa < ecolumn; ++fa) {
            for (uintt fb = brow; fb < erow; ++fb) {
                floatt retemp = 0;
                for (uintt fa1 = offset[0]; fa1 < offset[1]; ++fa1) {
                    uintt index = fa1 + realColumns1 * fb;
                    uintt index1 = fa1 * realColumns2 + fa;
                    retemp +=
                        -threadData->params[0].m_matrix->imValues[index] *
                        threadData->params[1].m_matrix->imValues[index1];
                }
                uintt index = fa + outputColumns * fb;
                threadData->outputs[0].m_matrix->reValues[index] = retemp;
            }
        }
    }
}
}
