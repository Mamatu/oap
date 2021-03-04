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

void TransposeOperationCpu::Execute(void* ptr) {
    ThreadData<TransposeOperationCpu>* threadData =
        (ThreadData<TransposeOperationCpu>*) ptr;
    intt bcolumn = threadData->begins[0];
    uintt brow = threadData->begins[1];
    uintt ecolumn = threadData->ends[0];
    uintt erow = threadData->ends[1];
    math::ComplexMatrix* m_output = threadData->outputs[0].m_matrix;
    const math::ComplexMatrix* m_matrix = threadData->params[0].m_matrix;
    uintt columns = gColumns (m_matrix);
    uintt columns1 = gColumns (m_output);
    if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
        threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = brow; fa < erow; fa++) {
            for (uintt fb = bcolumn; fb < ecolumn; fb++) {
                uintt index1 = fb * columns + fa;
                floatt value = gReValues (m_matrix)[index1];
                floatt value1 = gImValues (m_matrix)[index1];
                uintt indexa = fa * columns1 + fb;
                gReValues (m_output)[indexa] = value;
                gImValues (m_output)[indexa] = -value1;
            }
        }

    } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
        for (uintt fa = brow; fa < erow; fa++) {
            for (uintt fb = bcolumn; fb < ecolumn; fb++) {
                floatt value = gReValues (m_matrix)[fb * columns + fa];
                //gReValues (m_output)[fa * columns1 + fb] = value;
                SetRe(m_output, fb, fa, value);
            }
        }
    } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
        for (uintt fa = brow; fa < erow; fa++) {
            for (uintt fb = bcolumn; fb < ecolumn; fb++) {
                floatt value1 = gImValues (m_matrix)[fb * columns + fa];
                gImValues (m_output)[fa * columns1 + fb] = -value1;
            }
        }
    }
}

void TransposeOperationCpu::execute() {
    uintt threadsCount = oap::utils::mapper::createThreadsMap(getBMap(),
        this->m_threadsCount, gColumns (m_output) - m_subcolumns[0],
        gRows (m_output) - m_subrows[0]);
    ThreadData<TransposeOperationCpu>* threads = m_threadData;
    for (uintt fa = 0; fa < threadsCount; fa++) {
        threads[fa].outputs[0] = m_output;
        threads[fa].params[0] = m_matrix;
        threads[fa].calculateRanges(m_subcolumns, m_subrows, getBMap(), fa);
        threads[fa].thiz = this;
        threads[fa].thread.run (TransposeOperationCpu::Execute, &threads[fa]);
    }
    for (uint fa = 0; fa < threadsCount; fa++) {
        threads[fa].thread.stop();
    }
}
}
