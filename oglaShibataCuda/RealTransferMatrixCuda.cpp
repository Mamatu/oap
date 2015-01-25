/* 
 * File:   VirtualTransferMatrix.cpp
 * Author: mmatula
 * 
 * Created on January 13, 2014, 8:58 PM
 */

#include "HostMatrixModules.h"
#include "Matrix.h"
#include <math.h>
#include "RealTransferMatrixCuda.h"
#include "DeviceMatrixModules.h"
#include <vector>

namespace shibataCuda {
    inline void createUpIndecies(intt row, int columns, intt qc, intt* pows,
            int chainLength, intt** columnsIndecies, intt& columnsSize) {
        intt sp1 = row / pows[chainLength - 1];
        for (intt fa = 0; fa < columns; fa++) {
            if (fa % qc == sp1) {
                ArrayTools::add(columnsIndecies, columnsSize, fa);
            }
        }
    }

    inline void createDownIndecies(intt column, int rows, intt qc, intt* pows,
            int chainLength, intt** rowsIndecies, intt& rowsSize) {
        intt sp1 = column % qc;
        for (intt fa = 0; fa < rows; fa++) {
            if (fa / pows[chainLength - 1] == sp1) {
                ArrayTools::add(rowsIndecies, rowsSize, fa);
            }
        }
    }

    void increment(intt* array, intt length, intt max, bool istorus = false) {
        intt fa = 0;
        array[fa]++;
        while (true) {
            if (array[fa] >= max) {
                array[fa] = 0;
                if (fa >= length - 1) {
                    if (istorus) {
                        increment(array, length, max, istorus);
                    }
                    return;
                }
                fa++;
                array[fa]++;
            } else {
                return;
            }
        }
    }

    void increment(char* array, intt length, intt max, bool istorus = false) {
        intt fa = 0;
        array[fa]++;
        while (true) {
            if (array[fa] >= max) {
                array[fa] = 0;
                if (fa == length - 1) {
                    if (istorus) {
                        increment(array, length, max, istorus);
                    }
                    return;
                }
                fa++;
                array[fa]++;
            } else {
                return;
            }
        }
    }

    RealTransferMatrix::RealTransferMatrix() : ::shibataCpu::TransferMatrix() {
        tempSpins = NULL;
        mustBeDestroyed[0] = false;
        mustBeDestroyed[1] = false;
        mustBeDestroyed[2] = false;
        mustBeDestroyed[3] = false;
        m_isAllocated = false;
        m_output = NULL;
    }

    RealTransferMatrix::RealTransferMatrix(const RealTransferMatrix& orig) {
    }

    RealTransferMatrix::~RealTransferMatrix() {
        if (tempSpins != NULL) {
            delete[] tempSpins;
        }
        if (this->mustBeDestroyed[2]) {
            hmm.deleteMatrix(this->expHamiltonian1);
        }
        if (this->mustBeDestroyed[3]) {
            hmm.deleteMatrix(this->expHamiltonian2);
        }
        dealloc();
    }

    int RealTransferMatrix::getValueIndex(int index1, int columns, int index2) {
        return index1 + columns * index2;
    }

    int RealTransferMatrix::pow(int a, int b) {
        int o = 1;
        for (int fa = 0; fa < b; fa++) {
            o = o*a;
        }
        return o;
    }

    int RealTransferMatrix::getTMIndex(char** spinValueIndex, int posIndex, int timeLimit, int qc) {
        int index = 0;
        for (int fa = 0; fa < timeLimit; fa++) {
            int a = spinValueIndex[fa][posIndex];
            index += a * pow(qc, fa);
        }
        return index;
    }

    int RealTransferMatrix::getIndex(char** spinValueIndex, int posIndex, int timeIndex, int qc, int posLimit, int timeLimit) {
        char c1 = spinValueIndex[timeIndex][posIndex];
        int index = timeIndex + 1;
        if (index >= timeLimit) {
            index = 0;
        }
        char c2 = spinValueIndex[index][posIndex];
        return c1 + qc*c2;
    }

    bool RealTransferMatrix::isZero(std::vector<std::pair<int, int> >& vector, int index1, int index2) {
        for (int fa = 0; fa < vector.size(); fa++) {
            if (vector[fa].first == index1 && vector[fa].second == index2) {
                return true;
            }
        }
        return false;
    }

    int RealTransferMatrix::getIndex(char* spinValueIndex, int spin1Index, int qc, int count) {
        int output = 0;
        int spin2Index = spin1Index + 1;
        if (spin2Index == count) {
            spin2Index = 0;
        }
        output += spinValueIndex[spin1Index];
        output += qc * spinValueIndex[spin2Index];
        return output;
    }

    int RealTransferMatrix::getValue(int* matrix, int index) {
        return matrix[index];
    }

    int RealTransferMatrix::getSpinsCount() {
        return 3;
    }

    int RealTransferMatrix::getVirtualTime() {
        return this->parameters.getTrotterNumber()*2;
    }

    void RealTransferMatrix::transformMatrixOrientation(math::Matrix* dst,
            math::Matrix* src, shibataCpu::Orientation orientation) {
        if (orientation == shibataCpu::ORIENTATION_REAL_DIRECTION) {
            int spinsCount = src->rows;
            int qc = this->parameters.getQunatumsCount();
            tempSpins = tempSpins == NULL ? ArrayTools::create<char>(spinsCount, 0) : tempSpins;
            ArrayTools::clear<char>(tempSpins, spinsCount, 0);
            do {
                floatt rvalue = host::GetReValue(src, tempSpins[0] + qc * tempSpins[1], tempSpins[2] + qc * tempSpins[3]);
                floatt ivalue = host::GetImValue(src, tempSpins[0] + qc * tempSpins[1], tempSpins[2] + qc * tempSpins[3]);
                host::SetReValue(dst, tempSpins[2] + qc * tempSpins[0], tempSpins[3] + qc * tempSpins[1], rvalue);
                host::SetImValue(dst, tempSpins[2] + qc * tempSpins[0], tempSpins[3] + qc * tempSpins[1], ivalue);
            } while (ArrayTools::increment<char>(tempSpins, 0, qc, 1, 0, spinsCount) < spinsCount);
        }
    }

    math::Status RealTransferMatrix::onExecute() {
        return math::STATUS_OK;
    }

    inline void calculatePowers(intt a, intt b, intt* pows) {
        int o = 1;
        for (int fa = 0; fa < b; fa++) {
            pows[fa] = o;
            o = o*a;
        }
    }

    void RealTransferMatrix::alloc(::cuda::Kernel& kernel) {
        if (m_isAllocated == false) {
            int t[2];
            int b[2];
            DeviceMatrixUtils dmu;
            intt N = getSpinsCount();
            intt M2 = getVirtualTime();
            intt M = M2 / 2;
            assert("line below - check" == NULL);
            //kernel.getThreadsBlocks(t, b);
            int threadsCount = t[0] * t[1] * b[0] * b[1];
            intt width = t[0] * b[0];
            intt* pows = new intt[N * M2];
            intt quantumsCount = parameters.getQunatumsCount();
            calculatePowers(quantumsCount, N*M2, pows);

            intt* downIndecies = NULL;
            intt downIndeciesCount = 0;
            intt* upIndecies = NULL;
            intt upIndeciesCount = 0;
            intt nspins = quantumsCount;
            intt nvalues = nspins*quantumsCount;
            tmColumnsPtr = (intt**) CudaUtils::AllocDeviceMem(threadsCount * sizeof (intt*));
            tmRowsPtr = (intt**) CudaUtils::AllocDeviceMem(threadsCount * sizeof (intt*));
            tmRowsBitsPtr = (char**) CudaUtils::AllocDeviceMem(threadsCount * sizeof (char*));

            treePointers = (TreePointer***)
                    CudaUtils::AllocDeviceMem(threadsCount *
                    sizeof (TreePointer**));

            for (uintt fa = 0; fa < threadsCount; ++fa) {
                intt* tmColumns = (intt*) CudaUtils::AllocDeviceMem(M * sizeof (intt));
                intt* tmRows = (intt*) CudaUtils::AllocDeviceMem(M * sizeof (intt));
                char* tmRowsBits = (char*) CudaUtils::AllocDeviceMem(M2 * sizeof (char));
                CudaUtils::CopyHostToDevice(&tmColumnsPtr[fa], &tmColumns, sizeof (intt*));
                CudaUtils::CopyHostToDevice(&tmRowsPtr[fa], &tmRows, sizeof (intt*));
                CudaUtils::CopyHostToDevice(&tmRowsBitsPtr[fa], &tmRowsBits, sizeof (char*));

                TreePointer** treePointers1 = (TreePointer**) CudaUtils::AllocDeviceMem(M2 *
                        sizeof (TreePointer*));
                Pointers pointers = {tmColumns, tmRows, tmRowsBits, treePointers1};
                vpointers.push_back(pointers);
                for (intt fb = 0; fb < M2; ++fb) {
                    TreePointer* treePointer = m_treePointerCreator.create(fb,
                            expHamiltonian1,
                            expHamiltonian2);
                    treePointersVec.push_back(treePointer);
                    CudaUtils::CopyHostToDevice(&treePointers1[fb], &treePointer,
                            sizeof (TreePointer*));
                }
                CudaUtils::CopyHostToDevice(&treePointers[fa], &treePointers1,
                        sizeof (TreePointer**));
            }

            dupIndecies =
                    (intt*) CudaUtils::AllocDeviceMem(upIndeciesCount * sizeof (intt), upIndecies);
            ddownIndecies =
                    (intt*) CudaUtils::AllocDeviceMem(downIndeciesCount * sizeof (intt), downIndecies);
            ddownIndeciesCount = (intt*) CudaUtils::AllocDeviceObj(downIndeciesCount);
            dquantumsCount = (intt*) CudaUtils::AllocDeviceObj(quantumsCount);
            dM2 = (intt*) CudaUtils::AllocDeviceObj(M2);
            dwidth = (intt*) CudaUtils::AllocDeviceObj(width);
            uintt rows = dmu.getColumns(expHamiltonian1);
            uintt columns = dmu.getColumns(expHamiltonian2);
            for (intt fa = 0; fa < rows; fa++) {
                createDownIndecies(fa, columns, quantumsCount,
                        pows, 2, &downIndecies, downIndeciesCount);
            }
            for (intt fa = 0; fa < columns; fa++) {
                createUpIndecies(fa, rows, quantumsCount,
                        pows, 2, &upIndecies, upIndeciesCount);
            }
            m_isAllocated = true;
        }
    }

    void RealTransferMatrix::dealloc() {
        if (m_isAllocated == true) {
            CudaUtils::FreeDeviceObj(ddownIndeciesCount);
            CudaUtils::FreeDeviceObj(dquantumsCount);
            CudaUtils::FreeDeviceObj(dM2);
            for (uint fa = 0; fa < vpointers.size(); fa++) {
                CudaUtils::FreeDeviceMem(vpointers[fa].tmColumns);
                CudaUtils::FreeDeviceMem(vpointers[fa].tmRows);
                CudaUtils::FreeDeviceMem(vpointers[fa].tmRowsBits);
                CudaUtils::FreeDeviceMem(vpointers[fa].treePointers1);
            }
            CudaUtils::FreeDeviceMem(tmColumnsPtr);
            CudaUtils::FreeDeviceMem(tmRowsPtr);
            CudaUtils::FreeDeviceMem(tmRowsBitsPtr);
            for (uintt fa = 0; fa < treePointersVec.size(); fa++) {
                m_treePointerCreator.destroy(treePointersVec[fa]);
            }
            treePointersVec.clear();
        }
    }

    math::Status RealTransferMatrix::execute() {
        math::Status status = math::STATUS_OK;
        // Allocation and intitialization of tree pointers
        ::cuda::Kernel kernel;
        kernel.loadImage("liboglaShibataCuda.cubin");
        alloc(kernel);
        if (transferMatrix) {
            void* params[] = {&m_output, &treePointers, &tmColumnsPtr,
                &tmRowsPtr,
                &tmRowsBitsPtr,
                &dupIndecies, &ddownIndecies,
                &ddownIndeciesCount, &dquantumsCount, &dM2, &dwidth};
            kernel.setParams(params);
            kernel.loadImage("/home/mmatula/Ogla/oglaShibataCuda/\
dist/Debug/GNU-Linux-x86/liboglaShibataCuda.cubin");
            kernel.execute("ExecuteTM");
        } else {
            void* params[] = {
                &m_reoutputEntries,
                &m_imoutputEntries,
                &m_entries,
                &m_entriesCount,
                &treePointers,
                &tmColumnsPtr,
                &tmRowsPtr,
                &tmRowsBitsPtr,
                &dupIndecies,
                &ddownIndecies,
                &ddownIndeciesCount,
                &dquantumsCount,
                &dM2,
                &dwidth
            };
            kernel.setParams(params);
            kernel.loadImage("/home/mmatula/Ogla/oglaShibataCuda/\
dist/Debug/GNU-Linux-x86/liboglaShibataCuda.cubin");
            kernel.execute("ExecuteTM1");
        }
        return status;
    }
}