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

#define RE_DIAGONALIZE()\
if (threadData->params[1]->reValues[fa + columns * fb] != 0) {\
        floatt retemp = 0;\
        for (uint fa1 = 0; fa1 < columns; fa1++) {\
                retemp += threadData->params[0]->reValues[fa1 + columns * fb] * threadData->params[1]->reValues[fa1 * threadData->params[1]->columns + fa];\
        }\
        threadData->outputs[0]->reValues[fa + columns * fa] = retemp / threadData->params[1]->reValues[fa + columns * fb];\
}

#define IM_DIAGONALIZE()\
if (threadData->params[1]->imValues[fa + columns * fb] != 0) {\
        floatt imtemp = 0;\
        for (uint fa1 = 0; fa1 < columns; fa1++) {\
                imtemp += threadData->params[0]->imValues[fa1 + columns * fb] * threadData->params[1]->imValues[fa1 * threadData->params[1]->columns + fa];\
        }\
        threadData->outputs[0]->imValues[fa + columns * fa] = imtemp / threadData->params[1]->imValues[fa + columns * fb];\
}

namespace math {

    void DiagonalizationOperationCpu::Execute(void* ptr) {
        ThreadData<DiagonalizationOperationCpu>* threadData = (ThreadData<DiagonalizationOperationCpu>*) ptr;
        int bcolumn = threadData->begins[0];
        int ecolumn = threadData->ends[0];
        int columns = threadData->params[0]->columns;
        int rows = threadData->params[0]->rows;
        if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL && threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (uint fa = bcolumn; fa < ecolumn; fa++) {
                for (uint fb = 0; fb < rows; fb++) {
                    RE_DIAGONALIZE();
                    IM_DIAGONALIZE();
                }
            }
        } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
            for (uint fa = bcolumn; fa < ecolumn; fa++) {
                for (uint fb = 0; fb < rows; fb++) {
                    RE_DIAGONALIZE();
                }
            }
        } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
            for (uint fa = bcolumn; fa < ecolumn; fa++) {
                for (uint fb = 0; fb < rows; fb++) {
                    IM_DIAGONALIZE();
                }
            }
        }
    }

    void DiagonalizationOperationCpu::execute() {
        uintt* bmap = utils::mapper::allocMap(this->m_threadsCount);
        uintt threadsCount = utils::mapper::createThreadsMap(bmap,
                this->m_threadsCount, m_output->rows);
        ThreadData<DiagonalizationOperationCpu>* threads = new ThreadData<DiagonalizationOperationCpu>[threadsCount];
        for (intt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_output;
            threads[fa].params[0] = m_matrix1;
            threads[fa].params[1] = m_matrix2;
            threads[fa].calculateRanges(m_subcolumns, m_subrows, bmap, fa);
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(DiagonalizationOperationCpu::Execute, &threads[fa]);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uint fa = 0; fa < threadsCount; fa++) {

            threads[fa].thread.yield();
        }
        utils::mapper::freeMap(bmap);
        delete[] threads;
    }
}
