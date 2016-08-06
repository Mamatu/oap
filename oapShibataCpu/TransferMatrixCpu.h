/* 
 * File:   TransferMatrixCpu.h
 * Author: mmatula
 *
 * Created on January 27, 2014, 7:57 PM
 */

#ifndef OGLA_TRANSFER_MATRIX_CPU_H
#define	OGLA_TRANSFER_MATRIX_CPU_H

#include "TransferMatrix.h"
#include "TreePointerCreator.h"

namespace shibataCpu {
    namespace cpu {

        class Data {
        public:
            Data(uintt index);
            virtual ~Data();
            math::Matrix* transferMatrix;
            math::Matrix* expHamiltonian1;
            math::Matrix* expHamiltonian2;
            uintt m_index;
            floatt* m_reoutputEntries;
            floatt* m_imoutputEntries;
            uintt m_entriesCount;
            uintt* m_entries;
            shibataCpu::Parameters* parameters;
            int beginX;
            int beginY;
            int endX;
            int endY;
        };

        typedef void (*ExecuteTM_f)(Data* dataPtr, void* ptr);

        class TransferMatrix : public ::shibataCpu::TransferMatrix {
            HostMatrixAllocator hmm;
            TransferMatrix(const TransferMatrix& orig);
        public:
            TransferMatrix();
            virtual ~TransferMatrix();
            void setTransferMatrixExecution(ExecuteTM_f e, void* ptr);
        protected:
            virtual intt getThreadsCount() = 0;
            virtual int getSpinsChainsCount() = 0;
            virtual void onThreadCreation(uintt index, Data* data) = 0;
            math::Status execute();
        private:
            utils::mapper::ThreadsMap<uintt> m_map;
            static void Execute(void* ptr);
            static void Execute1(void* ptr);
            ExecuteTM_f executeTMFunction;
            void* ptr;
            uintt* bmap;
            uintt m_threadsCount1;
            void deallocRanges();
            void allocRanges(uintt threadsCount);
            int** begins;
            int** ends;

            class ThreadData : public Data {
                int spinsChainCount;
            public:
                void* ptr;
                ExecuteTM_f executeTM;
                ThreadData(uintt index);
                ~ThreadData();
                void setTransferMatrix(TransferMatrix* transferMatrix);
                TransferMatrix* thiz;
                int spinIndex;
                utils::Thread thread;
            };
            std::vector<ThreadData*> threads;
        };
    }
}
#endif	/* TRANSFERMATRIXCPU_H */

