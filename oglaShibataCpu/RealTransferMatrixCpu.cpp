/* 
 * File:   VirtualTransferMatrix.cpp
 * Author: mmatula
 * 
 * Created on January 13, 2014, 8:58 PM
 */

#include "HostMatrixModules.h"
#include "Matrix.h"
#include "Matrix.h"
#include "RealTransferMatrixCpu.h"
#include "TreePointer.h"

namespace shibata {
    namespace cpu {

        template<typename T>inline void conversion(T* output, uintt value, uint base) {
            for (uintt index = 0; value != 0; ++index) {
                output[index] = value % base;
                value = value / base;
            }
        }

        inline void createUpIndecies(uintt row, uintt columns, intt qc,
                intt* pows, int chainLength, uintt** indecies, uintt& size) {
            intt sp1 = row / pows[chainLength - 1];
            for (uintt fa = 0; fa < columns; fa++) {
                if (fa % qc == sp1) {
                    ArrayTools::add(indecies, size, fa);
                }
            }
        }

        inline void createDownIndecies(uintt column, uintt rows, intt qc,
                intt* pows, int chainLength, uintt** indecies, uintt& size) {
            intt sp1 = column % qc;
            for (uintt fa = 0; fa < rows; fa++) {
                if (fa / pows[chainLength - 1] == sp1) {
                    ArrayTools::add(indecies, size, fa);
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

        RealTransferMatrix::RealTransferMatrix() : ::shibata::cpu::TransferMatrix() {
            this->setTransferMatrixExecution(RealTransferMatrix::ExecuteTM, (void*) this);
            tempSpins = NULL;
            mustBeDestroyed[0] = false;
            mustBeDestroyed[1] = false;
            mustBeDestroyed[2] = false;
            mustBeDestroyed[3] = false;

            m_outputsEntries = NULL;
            m_entries = NULL;
            m_size = 0;
        }

        RealTransferMatrix::RealTransferMatrix(const RealTransferMatrix & orig) {
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
            m_outputsEntries = NULL;
            m_entries = NULL;
            m_size = 0;
        }

        int RealTransferMatrix::getValueIndex(int index1, int columns, int index2) {
            return index1 + columns * index2;
        }

        inline void calculatePowers(intt a, intt b, intt* pows) {
            int o = 1;
            for (int fa = 0; fa < b; fa++) {
                pows[fa] = o;
                o = o*a;
            }
        }

        int RealTransferMatrix::getSpinsChainsCount() {
            return 2;
        }

        int RealTransferMatrix::getSpinsCount() {
            return 3;
        }

        int RealTransferMatrix::getVirtualTime() {
            return this->parameters.getTrotterNumber()*2;
        }

        intt RealTransferMatrix::getThreadsCount() {
            return this->parameters.getThreadsCount();
        }

        void RealTransferMatrix::setThreadsCount(int threadsCount) {
            this->parameters.setThreadsCount(threadsCount);
        }

        math::Status RealTransferMatrix::onExecute() {
            return math::STATUS_OK;
        }

        void convertBitsToIndex(intt* rightRows, char* rightRowsBits, intt M) {
            intt index1 = 0;
            intt index2 = 0;
            for (intt fa = 0; fa < M; fa++) {
                index1 = fa * 2 + 1;
                index2 = fa * 2 + 2;
                if (index2 >= M * 2) {
                    index2 = 0;
                }
                rightRows[fa] = rightRowsBits[index1] +
                        rightRowsBits[index2]*2;
            }
        }

        inline floatt getReValue(math::Matrix* matrix, intt column, intt row) {
            if (matrix->reValues) {
                return matrix->reValues[row * matrix->columns + column];
            }
            return 0;
        }

        inline floatt getImValue(math::Matrix* matrix, intt column, intt row) {
            if (matrix->imValues) {
                return matrix->imValues[row * matrix->columns + column];
            }
            return 0;
        }

        inline bool isZero(math::Matrix* matrix, intt column, intt row) {
            bool is = (getReValue(matrix, column, row) == 0 &&
                    getImValue(matrix, column, row) == 0);
            return is;
        }

        inline floatt getReValue(TreePointer* previous) {
            if (previous->reValues) {
                return previous->reValues[previous->index];
            }
            return 0;
        }

        inline floatt getImValue(TreePointer* previous) {
            if (previous->imValues) {
                return previous->imValues[previous->index];
            }
            return 0;
        }

        inline intt getNodeValue(TreePointer* previous) {
            return previous->nodeValue[previous->index];
        }

        inline void SetNodeValue(TreePointer* previous, intt index, floatt revalue,
                floatt imvalue) {
            previous->nodeValue[previous->count] = index;
            previous->reValues[previous->count] = revalue;
            previous->imValues[previous->count] = imvalue;
            previous->count++;
        }

        inline void GetValue(TreePointer* current, intt index, intt* tmColumns,
                intt* tmRows, intt levelIndex, floatt* re, floatt* im) {
            floatt reValue = 0;
            floatt imValue = 0;
            intt tmindex = levelIndex / 2;
            if (current->type == TYPE_COLUMN) {
                reValue = getReValue(current->matrix, index, tmRows[tmindex]);
                imValue = getImValue(current->matrix, index, tmRows[tmindex]);
            } else {
                reValue = getReValue(current->matrix, tmColumns[tmindex], index);
                imValue = getImValue(current->matrix, tmColumns[tmindex], index);
            }
            (*re) = reValue;
            (*im) = imValue;
        }

        inline bool prepareLevel(intt levelIndex,
                TreePointer** treePointers, intt* tmColumns, intt* tmRows,
                uintt* upIndecies, uintt* downIndecies, intt count, intt qc) {
            TreePointer* previous = treePointers[levelIndex - 1];
            TreePointer* current = treePointers[levelIndex];
            current->count = 0;
            bool iszero = true;
            for (intt fa = 0; fa < qc; fa++) {
                const intt index1 = fa + qc * getNodeValue(previous);
                intt index = upIndecies[index1];
                floatt revalue = 0;
                floatt imvalue = 0;
                GetValue(current, index,
                        tmColumns, tmRows, levelIndex, &revalue, &imvalue);
                if (revalue != 0 || imvalue != 0) {
                    iszero = false;
                    floatt rev = getReValue(previous);
                    floatt imv = getImValue(previous);
                    floatt revalue1 = revalue * rev - imvalue * imv;
                    floatt imvalue1 = imvalue * rev + revalue * imv;
                    SetNodeValue(current, index, revalue1, imvalue1);
                }
            }
            return !iszero;
        }

        inline bool prepareFirstLevel(intt levelIndex,
                TreePointer** treePointers, intt* tmColumns, intt* tmRows,
                uintt* upIndecies, uintt* downIndecies, intt count, intt qc) {
            TreePointer* current = treePointers[levelIndex];
            current->count = 0;
            bool iszero = true;
            for (intt fa = 0; fa < count; fa++) {
                floatt revalue = 0;
                floatt imvalue = 0;
                GetValue(current, fa, tmColumns, tmRows, levelIndex,
                        &revalue,
                        &imvalue);
                if (revalue != 0 || imvalue != 0) {
                    iszero = false;
                    SetNodeValue(current, fa, revalue, imvalue);
                }
            }
            return iszero;
        }

        inline bool nextBranch(intt* levelIndex,
                TreePointer** treePointers,
                intt* tmColumns, intt* tmRows, bool p = false) {
            while (treePointers[*levelIndex]->index >=
                    treePointers[*levelIndex]->count) {
                treePointers[*levelIndex]->index = 0;
                (*levelIndex)--;
                if (*levelIndex < 0) {
                    return true;
                }
                treePointers[*levelIndex]->index++;
            }
            return false;
        }

        inline bool calculate(floatt* retm, floatt* imtm,
                intt x, intt y,
                intt* levelIndexPtr,
                TreePointer** treePointers,
                intt* tmColumns, intt* tmRows,
                uintt* upIndecies, uintt* downIndecies,
                intt count, intt qc) {
            TreePointer* previous = treePointers[(*levelIndexPtr) - 1];
            TreePointer* first = treePointers[0];
            TreePointer* current = treePointers[(*levelIndexPtr)];
            floatt revalue = getReValue(previous);
            floatt imvalue = getImValue(previous);
            for (intt fa = 0; fa < qc; fa++) {
                intt index = upIndecies[fa + qc * getNodeValue(previous)];
                for (intt fb = 0; fb < qc; fb++) {
                    intt index1 = downIndecies[fb + qc * getNodeValue(first)];
                    if (index == index1) {
                        floatt reValue1 = 0;
                        floatt imValue1 = 0;
                        GetValue(current, index,
                                tmColumns, tmRows, *levelIndexPtr, &reValue1, &imValue1);
                        if (reValue1 != 0 || imValue1 != 0) {
                            if (retm) {
                                floatt reV = *retm;
                                reV = reV + revalue * reValue1 - imvalue * imValue1;
                                *retm = reV;
                            }
                            if (imtm) {
                                floatt imV = *imtm;
                                imV = imV + revalue * imValue1 + imvalue * reValue1;
                                *imtm = imV;
                            }
                        }
                    }
                }
            }
            current->index = current->count;
            return nextBranch(levelIndexPtr, treePointers, tmColumns, tmRows);
        }

        RealTransferMatrix::TMData::TMData(TreePointerCreator*
                treePointerCreator) {
            M = 0;
            upIndeciesCount = 0;
            downIndeciesCount = 0;
            pows = NULL;
            tmColumns = NULL;
            tmRows = NULL;
            tmRowsBits = NULL;
            m_treePointerCreator = treePointerCreator;
            upIndecies = NULL;
            downIndecies = NULL;
        }

        void RealTransferMatrix::TMData::dealloc() {
            if (M != 0) {
                delete[] pows;
                delete[] tmColumns;
                delete[] tmRows;
                delete[] tmRowsBits;
                delete[] downIndecies;
                delete[] upIndecies;
                M = 0;
                for (intt fa = 0; fa < M * 2; fa++) {
                    m_treePointerCreator->destroy(treePointers[fa]);
                }
                delete[] treePointers;
            }
        }

        RealTransferMatrix::TMData::~TMData() {
            dealloc();
        }

        void RealTransferMatrix::TMData::alloc(uintt m, uintt N,
                uintt quantumsCount,
                math::Matrix* expHamiltonian1, math::Matrix* expHamiltonian2) {
            if (M != m) {
                if (M != 0) {
                    dealloc();
                }
                M = m;
                tmColumns = new intt[M];
                tmRows = new intt[M];
                tmRowsBits = new char[M * 2];
                pows = new intt[N * M * 2];
                calculatePowers(quantumsCount, N * M * 2, pows);
                for (uintt fa = 0; fa < expHamiltonian2->columns; fa++) {
                    createUpIndecies(fa, expHamiltonian1->rows, quantumsCount,
                            pows, 2, &upIndecies, upIndeciesCount);
                }
                for (uintt fa = 0; fa < expHamiltonian1->rows; fa++) {
                    createDownIndecies(fa, expHamiltonian2->columns, quantumsCount,
                            pows, 2, &downIndecies, downIndeciesCount);
                }
                treePointers = new TreePointer*[M * 2];
                for (intt fa = 0; fa < M * 2; fa++) {
                    treePointers[fa] = m_treePointerCreator->create(fa,
                            expHamiltonian1,
                            expHamiltonian2);
                }
            }
        }

        void RealTransferMatrix::onThreadCreation(uintt index,
                shibata::cpu::Data* data) {
            int M = data->parameters->getTrotterNumber();
            int N = data->parameters->getSpinsCount();
            uintt qc = data->parameters->getQunatumsCount();
            if (tmDatas.size() <= index) {
                TMData* tmData = new TMData(&m_treePointerCreator);
                tmData->alloc(M, N, qc, data->expHamiltonian1,
                        data->expHamiltonian2);
                tmDatas.push_back(tmData);
            }
        }

        void RealTransferMatrix::ExecuteTM(shibata::cpu::Data* dataPtr, void* ptr) {
            debugFunc();
            RealTransferMatrix* rtm = (RealTransferMatrix*) ptr;
            math::Matrix* transferMatrix = dataPtr->transferMatrix;
            shibata::Parameters* parameters = dataPtr->parameters;
            floatt* reoutputEntries = dataPtr->m_reoutputEntries;
            floatt* imoutputEntries = dataPtr->m_imoutputEntries;
            uintt* entries = dataPtr->m_entries;
            int N = rtm->getSpinsCount();
            int M2 = rtm->getVirtualTime();
            int M = M2 / 2;
            intt* pows = rtm->tmDatas[dataPtr->m_index]->pows;
            uintt quantumsCount = parameters->getQunatumsCount();
            intt* tmColumns = rtm->tmDatas[dataPtr->m_index]->tmColumns;
            intt* tmRows = rtm->tmDatas[dataPtr->m_index]->tmRows;
            char* tmRowsBits = rtm->tmDatas[dataPtr->m_index]->tmRowsBits;
            uintt upIndeciesCount = rtm->tmDatas[dataPtr->m_index]->upIndeciesCount;
            uintt* downIndecies = rtm->tmDatas[dataPtr->m_index]->downIndecies;
            uintt* upIndecies = rtm->tmDatas[dataPtr->m_index]->upIndecies;
            TreePointer** treePointers = rtm->tmDatas[dataPtr->m_index]->treePointers;
            intt nspins = quantumsCount;
            intt nvalues = nspins*quantumsCount;
            memset(tmRowsBits, 0, sizeof (char) * M2);
            memset(tmRows, 0, sizeof (intt) * M);
            memset(tmColumns, 0, sizeof (intt) * M);
            // Allocation and intitialization of tree pointers

            // Main execution
            for (uintt fa = dataPtr->beginX; fa < dataPtr->endX; fa++) {
                for (uintt fb = dataPtr->beginY; fb < dataPtr->endY; fb++) {
                    floatt* rev = NULL;
                    floatt* imv = NULL;
                    uintt column = 0;
                    uintt row = 0;
                    if (transferMatrix) {
                        uintt index = fa + transferMatrix->columns * fb;
                        column = fa;
                        row = fb;
                        if (transferMatrix->reValues) {
                            rev = &transferMatrix->reValues[index];
                        }
                        if (transferMatrix->imValues) {
                            imv = &transferMatrix->imValues[index];
                        }
                    } else {
                        uintt fa2 = fa * 2;
                        column = entries[fa2];
                        row = entries[fa2 + 1];
                        if (reoutputEntries) {
                            rev = &reoutputEntries[fa];
                        }
                        if (imoutputEntries) {
                            imv = &imoutputEntries[fa];
                        }
                    }
                    memset(tmRowsBits, 0, M2 * sizeof (char));
                    memset(tmRows, 0, M * sizeof (intt));
                    memset(tmColumns, 0, M * sizeof (intt));
                    conversion(tmRowsBits, row, 2);
                    conversion(tmColumns, column, nvalues);
                    intt levelIndex = 0;
                    bool finish = false;
                    convertBitsToIndex(tmRows, tmRowsBits, M);
                    while (finish == false) {
                        if (levelIndex == 0 &&
                                treePointers[levelIndex]->index == 0) {
                            finish = prepareFirstLevel(levelIndex, treePointers,
                                    tmColumns, tmRows,
                                    upIndecies, downIndecies,
                                    upIndeciesCount / quantumsCount, quantumsCount);
                            levelIndex++;
                        } else if (levelIndex == M2 - 1) {
                            finish = calculate(rev, imv, column, row,
                                    &levelIndex, treePointers,
                                    tmColumns, tmRows,
                                    upIndecies, downIndecies,
                                    upIndeciesCount / quantumsCount, quantumsCount);
                        } else {
                            if (levelIndex == 0) {
                                levelIndex++;
                            }
                            bool next = prepareLevel(levelIndex, treePointers,
                                    tmColumns, tmRows,
                                    upIndecies, downIndecies,
                                    upIndeciesCount / quantumsCount, quantumsCount);
                            if (next == false) {
                                finish = nextBranch(&levelIndex, treePointers,
                                        tmColumns, tmRows);
                            } else {
                                levelIndex++;
                            }
                        }
                    }
                    //increment(tmRowsBits, M2, 2, false);
                }
                //increment(tmColumns, M, nvalues, false);
            }
            //fprintf(stderr, "---------------\n");
            debugFunc();
        }
    }
}
