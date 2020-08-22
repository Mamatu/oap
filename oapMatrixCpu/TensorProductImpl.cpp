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

    void TensorProductOperationCpu::execute() {
        debugFuncBegin();
        uintt* bmap = utils::mapper::allocMap(this->m_threadsCount);
        uintt threadsCount = utils::mapper::createThreadsMap(bmap,
                this->m_threadsCount,
                gColumns (m_output),
                gRows (m_output));
        ThreadData<TensorProductOperationCpu>* threads =
                new ThreadData<TensorProductOperationCpu>[threadsCount];
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_output;
            threads[fa].params[0] = m_matrix1;
            threads[fa].params[1] = m_matrix2;
            threads[fa].calculateRanges(m_subcolumns, m_subrows, bmap, fa);
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(TensorProductOperationCpu::Execute,
                    &threads[fa]);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.join();
        }
        utils::mapper::freeMap(bmap);
        delete[] threads;
        debugFuncEnd();
    }

    void TensorProductOperationCpu::Execute(void* ptr) {
        ThreadData<TensorProductOperationCpu>* threadData = (ThreadData<TensorProductOperationCpu>*) ptr;
        int begin1 = threadData->begins[0];
        int end1 = threadData->ends[0];
        int begin2 = threadData->begins[1];
        int end2 = threadData->ends[1];
        const intt columns = gColumns (threadData->outputs[0].m_matrix);
        const intt columns1 = gColumns (threadData->params[0].m_matrix);
        const intt columns2 = gColumns (threadData->params[1].m_matrix);
        const intt c1 = gColumns (threadData->params[0].m_matrix);
        const intt c2 = gColumns (threadData->params[1].m_matrix);
        if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa = begin1; fa < end1; fa++) {
                for (intt fb = begin2; fb < end2; fb++) {
                    int fa1 = fa / c1;
                    int fa2 = fa % c2;
                    int fb1 = fb / c1;
                    int fb2 = fb % c2;
                    int index2 = (fa + columns * fb);
                    int index1 = (fa1 + columns1 * fb1);
                    int index = (fa2 + columns2 * fb2);
                    *GetRePtrIndex (threadData->outputs[0].m_matrix, index2) =
                            GetReIndex (threadData->params[0].m_matrix, index) *
                            GetReIndex (threadData->params[1].m_matrix, index1) -
                            GetImIndex (threadData->params[0].m_matrix, index) *
                            GetImIndex (threadData->params[1].m_matrix, index1);
                    *GetImPtrIndex (threadData->outputs[0].m_matrix, index2) =
                            GetReIndex (threadData->params[0].m_matrix, index) *
                            GetImIndex (threadData->params[1].m_matrix, index1) -
                            GetImIndex (threadData->params[0].m_matrix, index) *
                            GetReIndex (threadData->params[1].m_matrix, index1);
                }
            }
        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (intt fa = begin1; fa < end1; fa++) {
                for (intt fb = begin2; fb < end2; fb++) {
                    int fa1 = fa / c1;
                    int fa2 = fa % c2;
                    int fb1 = fb / c1;
                    int fb2 = fb % c2;
                    int index2 = (fa + columns * fb);
                    int index1 = (fa1 + columns1 * fb1);
                    int index = (fa2 + columns2 * fb2);
                    *GetRePtrIndex (threadData->outputs[0].m_matrix, index2) =
                            GetReIndex (threadData->params[0].m_matrix, index) *
                            GetReIndex (threadData->params[1].m_matrix, index1);
                }
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa = begin1; fa < end1; fa++) {
                for (intt fb = begin2; fb < end2; fb++) {
                    int fa1 = fa / c1;
                    int fa2 = fa % c2;
                    int fb1 = fb / c1;
                    int fb2 = fb % c2;
                    int index2 = (fa + columns * fb);
                    int index1 = (fa1 + columns1 * fb1);
                    int index = (fa2 + columns2 * fb2);
                    *GetRePtrIndex (threadData->outputs[0].m_matrix, index2) =
                            -GetImIndex (threadData->params[0].m_matrix, index) *
                            GetImIndex (threadData->params[1].m_matrix, index1);
                }
            }
        }
    }
}
