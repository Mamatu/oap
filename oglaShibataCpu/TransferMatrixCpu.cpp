/* 
 * File:   TransferMatrix.cpp
 * Author: mmatula
 * 
 * Created on January 27, 2014, 7:57 PM
 */

#include <vector>

#include "TransferMatrixCpu.h"
#include "ThreadsMapper.h"

namespace shibata {
    namespace cpu {

        Data::Data(uintt index) {
            m_index = index;
            transferMatrix = NULL;
            expHamiltonian1 = NULL;
            expHamiltonian2 = NULL;
            m_reoutputEntries = NULL;
            m_imoutputEntries = NULL;
            m_entriesCount = 0;
            m_entries = NULL;
            parameters = NULL;
            beginX = 0;
            beginY = 0;
            endX = 0;
            endY = 0;
        }

        Data::~Data() {
        }

        class ThreadData1 {
        public:

            ThreadData1() : hamiltonian(NULL), expHamiltonian(NULL) {
            }
            math::Matrix* hamiltonian;
            math::Matrix* expHamiltonian;
            int trotterNumber;
            utils::Thread thread;
        };

        void createThreadData(ThreadData1& td, math::Matrix* hamiltonian,
                math::Matrix* expHamiltonian,
                HostMatrixAllocator& mhm, int trotterNumber) {
            if (expHamiltonian == NULL) {
                td.hamiltonian = hamiltonian;
                td.expHamiltonian = host::NewMatrixCopy(hamiltonian);
            } else {
                td.hamiltonian = NULL;
                td.expHamiltonian = expHamiltonian;
            }
            td.trotterNumber = trotterNumber;
        }

        void destroyThreadData(ThreadData1& td, HostMatrixAllocator& mhm,
                bool destroyExpHamiltonian) {
            if (destroyExpHamiltonian == true) {
                mhm.deleteMatrix(td.expHamiltonian);
            }
        }

        TransferMatrix::TransferMatrix() : shibata::TransferMatrix() {
            bmap = NULL;
            m_threadsCount1 = 0;
            begins = NULL;
            ends = NULL;
        }

        TransferMatrix::TransferMatrix(const TransferMatrix& orig) {
        }

        TransferMatrix::~TransferMatrix() {
            if (bmap) {
                utils::mapper::freeMap(bmap);
            }
            for (uintt fa = 0; fa < threads.size(); ++fa) {
                delete threads[fa];
            }
            deallocRanges();
            threads.clear();
        }

        TransferMatrix::ThreadData::ThreadData(uintt index) :
        Data(index), executeTM(NULL), thiz(NULL) {
        }

        void TransferMatrix::ThreadData::setTransferMatrix(TransferMatrix* transferMatrix) {
            thiz = transferMatrix;
            int virtualTimeCount = thiz->getVirtualTime();
            int spinsCount = thiz->getSpinsCount();
            spinsChainCount = thiz->getSpinsChainsCount();
            debug("VT count: %d \n", virtualTimeCount);
            debug("Spins count: %d \n", spinsCount);
            this->ptr = thiz->ptr;
            spinIndex = 0;
        }

        void TransferMatrix::setTransferMatrixExecution(ExecuteTM_f executeTM, void* ptr) {
            this->executeTMFunction = executeTM;
            this->ptr = ptr;
        }

        void TransferMatrix::Execute(void* ptr) {
            TransferMatrix::ThreadData* threadData = (TransferMatrix::ThreadData*)ptr;
            threadData->executeTM(threadData, threadData->ptr);
        }

        void TransferMatrix::Execute1(void* ptr) {
            HostMatrixUtils mu;
            HostMatrixPrinter hmp;
            ThreadData1* threadData1 = (ThreadData1*) ptr;
            if (threadData1->hamiltonian) {
                math::ExpOperationCpu expOperation;
                math::MultiplicationConstOperationCpu mco;
                mco.setMatrix(threadData1->hamiltonian);
                floatt factor = -1. / threadData1->trotterNumber;
                mco.setReValue(&factor);
                mco.setOutputMatrix(threadData1->hamiltonian);
                mco.start();
                expOperation.setMatrix(threadData1->hamiltonian);
                expOperation.setOutputMatrix(threadData1->expHamiltonian);
            }
            hmp.printReMatrix(threadData1->expHamiltonian);
        }

        void TransferMatrix::deallocRanges() {
            if (begins) {
                for (int fa = 0; fa < m_threadsCount1; fa++) {
                    delete[] begins[fa];
                    delete[] ends[fa];
                }
                delete[] begins;
                delete[] ends;
                begins = NULL;
                ends = NULL;
            }
        }

        void TransferMatrix::allocRanges(uintt threadsCount) {
            begins = new int*[threadsCount];
            ends = new int*[threadsCount];
            for (int fa = 0; fa < threadsCount; fa++) {
                begins[fa] = new int[2];
                ends[fa] = new int[2];
            }
        }

        math::Status TransferMatrix::execute() {
            math::Status code = this->onExecute();
            if (code != math::STATUS_OK) {
                return code;
            }
            if (this->expHamiltonian1 == NULL || this->expHamiltonian2 == NULL
                    || (this->transferMatrix == NULL &&
                    ((m_reoutputEntries == NULL && m_imoutputEntries == NULL)
                    || m_entries == NULL))) {
                return math::STATUS_INVALID_PARAMS;
            }
            if (this->parameters.areAllAvailable() == false) {
                return math::STATUS_ERROR;
            }
            bool destroyExpHamiltonian = false;
            if (this->expHamiltonian1 == NULL || this->expHamiltonian2 == NULL) {
                destroyExpHamiltonian = true;
            }
            uintt m_threadsCount = this->getThreadsCount();
            if (bmap || m_threadsCount1 != m_threadsCount) {
                if (bmap) {
                    utils::mapper::freeMap(bmap);
                    deallocRanges();
                }
                bmap = utils::mapper::allocMap(m_threadsCount);
                m_threadsCount1 = m_threadsCount;
                allocRanges(m_threadsCount);
            }

            uintt width = 0;
            uintt height = 0;
            if (this->transferMatrix) {
                width = this->transferMatrix->columns;
                height = this->transferMatrix->rows;
            } else if (m_reoutputEntries) {
                height = 1;
                width = m_entriesCount;
            }
            uintt threadsCount = utils::mapper::createThreadsMap(bmap,
                    m_threadsCount,
                    width,
                    height);

            uintt size = threads.size();
            if (threadsCount > size) {
                for (uintt fa = size; fa < threadsCount; ++fa) {
                    TransferMatrix::ThreadData* threadData =
                            new TransferMatrix::ThreadData(fa);
                    threads.push_back(threadData);
                }
            }

            for (uintt fa = 0; fa < threadsCount; ++fa) {
                TransferMatrix::ThreadData* threadData = threads[fa];
                utils::mapper::getThreadsMap(m_map, bmap, fa);
                threadData->transferMatrix = this->transferMatrix;
                threadData->expHamiltonian1 = expHamiltonian1;
                threadData->expHamiltonian2 = expHamiltonian2;
                threadData->m_reoutputEntries = m_reoutputEntries;
                threadData->m_imoutputEntries = m_imoutputEntries;
                threadData->m_entriesCount = m_entriesCount;
                threadData->m_entries = m_entries;
                threadData->parameters = &this->parameters;
                threadData->executeTM = this->executeTMFunction;
                threadData->thiz = this;
                threadData->beginX = m_map.beginColumn;
                threadData->endX = m_map.endColumn;
                threadData->beginY = m_map.beginRow;
                threadData->endY = m_map.endRow;
                threadData->setTransferMatrix(this);
                threadData->thread.setFunction(TransferMatrix::Execute, threadData);
            }

            if (threadsCount > size) {
                for (uintt fa = size; fa < threadsCount; ++fa) {
                    TransferMatrix::ThreadData* threadData = threads[fa];
                    onThreadCreation(fa, threadData);
                }
            }
            for (uintt fa = 0; fa < threadsCount; ++fa) {
                TransferMatrix::ThreadData* threadData = threads[fa];
                threadData->thread.run();
            }
            debugFunc();
            for (uint fa = 0; fa < threadsCount; fa++) {
                threads[fa]->thread.yield();
            }
            debugFunc();
            return math::STATUS_OK;
        }

        TransferMatrix::ThreadData::~ThreadData() {
        }
    }
}