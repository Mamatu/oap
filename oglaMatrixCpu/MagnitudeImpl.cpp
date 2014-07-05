#include "MathOperationsCpu.h"
#include "Internal.h"
#include <math.h>
namespace math {
    namespace cpu {

        void MagnitudeOperation::Execute(void* ptr) {
            ThreadData<MagnitudeOperation>* threadData = (ThreadData<MagnitudeOperation>*) ptr;
            floatt output = 0;
            if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL &&
                    threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
                for (intt fa = threadData->begins[0]; fa < threadData->ends[0]; fa++) {
                    floatt v1 = threadData->params[0]->m_matrix->reValues[fa] * threadData->params[0]->m_matrix->reValues[fa];
                    floatt v2 = threadData->params[0]->m_matrix->imValues[fa] * threadData->params[0]->m_matrix->imValues[fa];
                    output += v1 + v2;
                }
            } else if (threadData->thiz->m_executionPathRe == EXECUTION_NORMAL) {
                for (intt fa = threadData->begins[0]; fa < threadData->ends[0]; fa++) {
                    output += threadData->params[0]->m_matrix->reValues[fa] *
                            threadData->params[0]->m_matrix->reValues[fa];
                }
            } else if (threadData->thiz->m_executionPathIm == EXECUTION_NORMAL) {
                for (intt fa = threadData->begins[0]; fa < threadData->ends[0]; fa++) {
                    output += threadData->params[0]->m_matrix->imValues[fa] *
                            threadData->params[0]->m_matrix->imValues[fa];
                }
            }
            threadData->values[0] = output;
        }

        void MagnitudeOperation::execute() {
            uintt* bmap = utils::mapper::allocMap(this->m_threadsCount);
            uintt length = m_matrixStructure->m_subcolumns * m_matrixStructure->m_subrows;
            uintt threadsCount = utils::mapper::createThreadsMap(bmap, this->m_threadsCount, length);
            ThreadData<MagnitudeOperation>* threads = new ThreadData<MagnitudeOperation>[threadsCount];
            for (intt fa = 0; fa < threadsCount; fa++) {
                threads[fa].params[0] = m_matrixStructure;
                threads[fa].calculateRanges(m_matrixStructure, bmap, fa);
                threads[fa].thiz = this;
                threads[fa].thread.setFunction(MagnitudeOperation::Execute, &threads[fa]);
                threads[fa].thread.run((this->m_threadsCount == 1));
            }
            for (uint fa = 0; fa < threadsCount; fa++) {
                threads[fa].thread.yield();
            }
            floatt output = 0.;
            for (uint fa = 0; fa < threadsCount; fa++) {
                output += threads[fa].values[0];
            }
            (*this->m_output1) = sqrt(output);
            utils::mapper::freeMap(bmap);
            delete[] threads;
        }
    }
}
