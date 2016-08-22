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




#ifndef OAP_TRANSFER_MATRIX_CPU_H
#define	OAP_TRANSFER_MATRIX_CPU_H

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
