#include "MathOperationsCpu.h"
#include "Internal.h"
namespace math {

    void TensorProductOperationCpu::execute() {
        debugFuncBegin();
        uintt* bmap = utils::mapper::allocMap(this->m_threadsCount);
        uintt threadsCount = utils::mapper::createThreadsMap(bmap,
                this->m_threadsCount,
                m_output->columns,
                m_output->rows);
        ThreadData<TensorProductOperationCpu>* threads =
                new ThreadData<TensorProductOperationCpu>[threadsCount];
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].outputs[0] = m_output;
            threads[fa].params[0] = m_matrix1;
            threads[fa].params[1] = m_matrix2;
            threads[fa].calculateRanges(m_output, bmap, fa);
            threads[fa].thiz = this;
            threads[fa].thread.setFunction(TensorProductOperationCpu::Execute,
                    &threads[fa]);
            threads[fa].thread.run((this->m_threadsCount == 1));
        }
        for (uintt fa = 0; fa < threadsCount; fa++) {
            threads[fa].thread.yield();
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
        const intt columns = threadData->outputs[0]->columns;
        const intt columns1 = threadData->params[0]->columns;
        const intt columns2 = threadData->params[1]->columns;
        const intt c1 = threadData->params[0]->columns;
        const intt c2 = threadData->params[1]->columns;
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
                    threadData->outputs[0]->reValues[index2] =
                            threadData->params[0]->reValues[index] *
                            threadData->params[1]->reValues[index1] -
                            threadData->params[0]->imValues[index] *
                            threadData->params[1]->imValues[index1];
                    threadData->outputs[0]->imValues[index2] =
                            threadData->params[0]->reValues[index] *
                            threadData->params[1]->imValues[index1] -
                            threadData->params[0]->imValues[index] *
                            threadData->params[1]->reValues[index1];
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
                    threadData->outputs[0]->reValues[index2] =
                            threadData->params[0]->reValues[index] *
                            threadData->params[1]->reValues[index1];
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
                    threadData->outputs[0]->reValues[index2] =
                            -threadData->params[0]->imValues[index] *
                            threadData->params[1]->imValues[index1];
                }
            }
        }
    }
}