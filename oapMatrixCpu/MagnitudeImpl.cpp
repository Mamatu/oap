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
#include <math.h>
namespace math {

    void MagnitudeOperationCpu::Execute(void* ptr) {
        ThreadData<MagnitudeOperationCpu>* threadData = (ThreadData<MagnitudeOperationCpu>*) ptr;
        floatt output = 0;
        if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa = threadData->begins[0]; fa < threadData->ends[0]; fa++) {
                floatt v1 = GetReIndex (threadData->params[0].m_matrix, fa) *
                        GetReIndex (threadData->params[0].m_matrix, fa);
                floatt v2 = GetImIndex (threadData->params[0].m_matrix, fa) *
                        GetImIndex (threadData->params[0].m_matrix, fa);
                output += v1 + v2;
            }
        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (intt fa = threadData->begins[0]; fa < threadData->ends[0]; fa++) {
                output += GetReIndex (threadData->params[0].m_matrix, fa) *
                        GetReIndex (threadData->params[0].m_matrix, fa);
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (intt fa = threadData->begins[0]; fa < threadData->ends[0]; fa++) {
                output += GetImIndex (threadData->params[0].m_matrix, fa) *
                        GetImIndex (threadData->params[0].m_matrix, fa);
            }
        }
        threadData->values[0] = output;
    }

    void MagnitudeOperationCpu::execute() {
        uintt* bmap = oap::utils::mapper::allocMap(this->m_threadsCount);
        uintt length = gColumns (m_matrix) * gRows (m_matrix);
        uintt threadsCount = oap::utils::mapper::createThreadsMap(bmap, this->m_threadsCount, length);
        ThreadData<MagnitudeOperationCpu>* threads = new ThreadData<MagnitudeOperationCpu>[threadsCount];
        for (intt fa = 0; fa < threadsCount; fa++) {
            threads[fa].params[0] = m_matrix;
            threads[fa].calculateRanges(m_subcolumns, m_subrows, bmap, fa);
            threads[fa].thiz = this;
            threads[fa].thread.run(MagnitudeOperationCpu::Execute, &threads[fa]);
        }
        for (uint fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.stop();
        }
        floatt output = 0.;
        for (uint fa = 0; fa < threadsCount; fa++) {
            output += threads[fa].values[0];
        }
        (*this->m_output1) = sqrt(output);
        oap::utils::mapper::freeMap(bmap);
        delete[] threads;
    }
}
