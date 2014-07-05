/* 
 * File:   VirtualTransferMatrix.h
 * Author: mmatula
 *
 * Created on January 13, 2014, 8:58 PM
 */

#ifndef OGLA_VIRTUALTRANSFER_MATRIX_CUDA_H
#define	OGLA_VIRTUALTRANSFER_MATRIX_CUDA_H

#include "MatrixStructure.h"
#include "TransferMatrix.h"
#include "DeviceTreePointerCreator.h"
#include "KernelExecutor.h"

namespace shibata {
    namespace cuda {

        class RealTransferMatrix : public shibata::TransferMatrix {
        public:
            RealTransferMatrix();
            virtual ~RealTransferMatrix();
        protected:
            DeviceTreePointerCreator m_treePointerCreator;
            bool mustBeDestroyed[4];
            int getSpinsCount();
            int getVirtualTime();
            math::Status onExecute();
            math::Status execute();
        private:

            struct Pointers {
                intt* tmColumns;
                intt* tmRows;
                char* tmRowsBits;
                TreePointer** treePointers1;
            };
            bool m_isAllocated;
            void alloc(::cuda::KernelMatrix& kernel);
            void dealloc();

            intt** tmColumnsPtr;
            intt** tmRowsPtr;
            char** tmRowsBitsPtr;
            TreePointer*** treePointers;
            intt* dupIndecies;
            intt* ddownIndecies;
            intt* ddownIndeciesCount;
            intt* dquantumsCount;
            intt* dM2;
            intt* dwidth;
            MatrixStructure* m_outputStructure;
            std::vector<Pointers> vpointers;
            std::vector<TreePointer*> treePointersVec;
            HostMatrixUtils hmu;
            HostMatrixPrinter hmp;
            char* tempSpins;
            inline int getValueIndex(int index1, int columns, int index2);
            inline int pow(int a, int b);
            inline int getTMIndex(char** spinValueIndex, int posIndex, int timeLimit, int qc);
            inline int getIndex(char** spinValueIndex, int posIndex, int timeIndex, int qc, int posLimit, int timeLimit);
            inline bool isZero(std::vector<std::pair<int, int> >& vector, int index1, int index2);
            inline int getIndex(char* spinValueIndex, int spin1Index, int qc, int count);
            inline int getValue(int* matrix, int index);
            void transformMatrixOrientation(math::Matrix* dst, math::Matrix* src, Orientation orientation);
            HostMatrixAllocator hmm;
            RealTransferMatrix(const RealTransferMatrix& orig);
        };
    }
}

#endif	/* VIRTUALTRANSFERMATRIX_H */

