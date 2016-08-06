#include <cuda.h>
#include "TreePointer.h"
#include <stdio.h>

__device__ floatt getReValue(math::Matrix* matrix, intt column, intt row) {
    if (matrix->reValues) {
        uintt index = row * matrix->columns + column;
        return matrix->reValues[index];
    }
    return 0;
}

__device__ floatt getImValue(math::Matrix* matrix, intt column, intt row) {
    if (matrix->imValues) {
        uintt index = row * matrix->columns + column;
        return matrix->imValues[index];
    }
    return 0;
}

__device__ bool isZero(math::Matrix* matrix, intt column, intt row) {
    bool is = (getReValue(matrix, column, row) == 0);
    return is;
}

__device__ floatt GetReValue(math::Matrix* tm, uintt x, uintt y) {
    if (tm->reValues) {
        return tm->reValues[x + tm->columns * y];
    }
    return 0;
}

__device__ floatt GetImValue(math::Matrix* tm, uintt x, uintt y) {
    if (tm->imValues) {
        return tm->imValues[x + tm->columns * y];
    }
    return 0;
}

__device__ void SetReValue(math::Matrix* tm, uintt x, uintt y,
        floatt v) {
    if (tm->reValues) {
        tm->reValues[x + tm->columns * y] = v;
    }
}

__device__ void SetImValue(math::Matrix* tm, uintt x, uintt y,
        floatt v) {
    if (tm->imValues) {
        tm->imValues[x + tm->columns * y] = v;
    }
}

__device__ floatt getReValue(TreePointer* previous) {
    if (previous->reValues) {
        return previous->reValues[previous->index];
    }
    return 0;
}

__device__ floatt getImValue(TreePointer* previous) {
    if (previous->imValues) {
        return previous->imValues[previous->index];
    }
    return 0;
}

__device__ intt getNodeValue(TreePointer* previous) {
    return previous->nodeValue[previous->index];
}

__device__ void SetNodeValue(TreePointer* previous,
        intt index, floatt reValue, floatt imValue) {
    previous->nodeValue[previous->count] = index;
    previous->reValues[previous->count] = reValue;
    previous->imValues[previous->count] = imValue;
    previous->count++;
}

__device__ void GetValue(TreePointer* current,
        intt index, intt* tmColumns,
        intt* tmRows, intt levelIndex,
        floatt* re, floatt* im) {
    floatt revalue = 0;
    floatt imvalue = 0;
    intt tmindex = levelIndex / 2;
    if (current->type == TYPE_COLUMN) {
        revalue = getReValue(current->matrix, index, tmRows[tmindex]);
        imvalue = getImValue(current->matrix, index, tmRows[tmindex]);
    } else {
        revalue = getReValue(current->matrix, tmColumns[tmindex], index);
        imvalue = getImValue(current->matrix, tmColumns[tmindex], index);
    }
    *re = revalue;
    *im = imvalue;
}

__device__ bool prepareLevel(intt levelIndex,
        TreePointer** treePointers, intt* tmColumns, intt* tmRows,
        intt* upIndecies, intt* downIndecies, intt count, intt qc) {
    TreePointer* previous = treePointers[levelIndex - 1];
    TreePointer* current = treePointers[levelIndex];
    current->count = 0;
    bool iszero = true;
    for (intt fa = 0; fa < qc; fa++) {
        const intt index1 = fa + qc * getNodeValue(previous);
        intt index = upIndecies[index1];
        floatt revalue = 0;
        floatt imvalue = 0;
        GetValue(current, index, tmColumns, tmRows, levelIndex,
                &revalue, &imvalue);
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

__device__ bool prepareFirstLevel(intt levelIndex,
        TreePointer** treePointers, intt* tmColumns, intt* tmRows,
        intt* upIndecies, intt* downIndecies, intt count, intt qc) {
    TreePointer* current = treePointers[levelIndex];
    current->count = 0;
    bool iszero = true;
    for (intt fa = 0; fa < count; ++fa) {
        floatt revalue = 0;
        floatt imvalue = 0;
        GetValue(current, fa, tmColumns, tmRows, levelIndex, &revalue, &imvalue);
        if (revalue != 0 || imvalue != 0) {
            iszero = false;
            SetNodeValue(current, fa, revalue, imvalue);
        }
    }
    return iszero;
}

__device__ bool nextBranch(intt* levelIndex,
        TreePointer** treePointers,
        intt* tmColumns, intt* tmRows) {
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

__device__ bool calculate(floatt* re, floatt* im, intt x, intt y,
        intt* levelIndexPtr,
        TreePointer** treePointers,
        intt* tmColumns, intt* tmRows,
        intt* upIndecies, intt* downIndecies, intt count, intt qc) {
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
                floatt revalue1 = 0;
                floatt imvalue1 = 0;
                GetValue(current, index,
                        tmColumns, tmRows, *levelIndexPtr,
                        &revalue1, &imvalue1);
                if (revalue1 != 0 || imvalue1 != 0) {
                    floatt rev = *re;
                    floatt imv = *im;
                    rev = rev + revalue * revalue1 - imvalue * imvalue1;
                    imv = imv + revalue * imvalue1 + imvalue * revalue1;
                    *re = rev;
                    *im = imv;
                }
            }
        }
    }
    current->index = current->count;
    return nextBranch(levelIndexPtr, treePointers, tmColumns, tmRows);
}

__device__ bool calculate(math::Matrix* tm, intt x, intt y,
        intt* levelIndexPtr,
        TreePointer** treePointers,
        intt* tmColumns, intt* tmRows,
        intt* upIndecies, intt* downIndecies, intt count, intt qc) {
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
                floatt revalue1 = 0;
                floatt imvalue1 = 0;
                GetValue(current, index,
                        tmColumns, tmRows, *levelIndexPtr,
                        &revalue1, &imvalue1);
                if (revalue1 != 0 || imvalue1 != 0) {
                    floatt rev = GetReValue(tm, x, y);
                    floatt imv = GetImValue(tm, x, y);
                    rev = rev + revalue * revalue1 - imvalue * imvalue1;
                    imv = imv + revalue * imvalue1 + imvalue * revalue1;
                    SetReValue(tm, x, y, rev);
                    SetImValue(tm, x, y, imv);
                }
            }
        }
    }
    current->index = current->count;
    return nextBranch(levelIndexPtr, treePointers, tmColumns, tmRows);
}

__device__ void increment(intt* array, intt length, intt max, bool istorus = false) {
    intt fa = 0;
    array[fa]++;
    while (true) {
        if (array[fa] >= max) {
            array[fa] = 0;
            if (fa >= length - 1) {
                return;
            }
            fa++;
            array[fa]++;
        } else {
            return;
        }
    }
}

__device__ void increment(char* array, intt length, intt max, bool istorus = false) {
    intt fa = 0;
    array[fa]++;
    while (true) {
        if (array[fa] >= max) {
            array[fa] = 0;
            if (fa == length - 1) {
                return;
            }
            fa++;
            array[fa]++;
        } else {
            return;
        }
    }
}

__device__ void convertBitsToIndex(intt* rightRows, char* rightRowsBits, intt M) {
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

extern "C" __global__ void ExecuteTM1(
        floatt* reoutpus,
        floatt* imoutpus,
        uintt* entries,
        uintt count,
        TreePointer*** treePointers1,
        intt** tmColumns1,
        intt** tmRows1, char** tmRowsBits1,
        intt* upIndecies, intt* downIndecies,
        intt* columnsIndeciesCountPtr,
        intt* quantumsCountPtr, intt* M2Ptr, intt* width) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    uintt index = threadIndexX + *width * threadIndexY;
    TreePointer** treePointers = treePointers1[index];

    intt* tmColumns = tmColumns1[index];
    intt* tmRows = tmRows1[index];
    char* tmRowsBits = tmRowsBits1[index];
    uintt column = entries[threadIndexX * 2];
    uintt row = entries[threadIndexX * 2 + 1];
    floatt* re = &reoutpus[threadIndexX];
    floatt* im = &imoutpus[threadIndexX];
    intt M2 = *M2Ptr;
    uintt nvalues = *quantumsCountPtr * *quantumsCountPtr;

    for (uintt fb = 0; fb < M2; ++fb) {
        tmRowsBits[fb] = 0;
    }

    for (uintt fb = 0; fb < M2 / 2; ++fb) {
        tmRows[fb] = 0;
        tmColumns[fb] = 0;
    }

    for (uintt fb = 0; fb < row; ++fb) {
        increment(tmRowsBits, M2, 2, false);
    }
    for (uintt fa = 0; fa < column; ++fa) {
        increment(tmColumns, M2 / 2, nvalues, false);
    }
    convertBitsToIndex(tmRows, tmRowsBits, M2 / 2);
    uintt fa = threadIndexX;
    uintt fb = threadIndexY;
    uintt columnsIndeciesCount = *columnsIndeciesCountPtr;
    intt quantumsCount = *quantumsCountPtr;
    bool finish = false;
    intt levelIndex = 0;

    while (finish == false) {
        if (levelIndex == 0 && treePointers[levelIndex]->index == 0) {
            finish = prepareFirstLevel(levelIndex, treePointers,
                    tmColumns, tmRows,
                    upIndecies, downIndecies,
                    columnsIndeciesCount / quantumsCount, quantumsCount);
            levelIndex++;
        } else if (levelIndex == M2 - 1) {
            finish = calculate(re, im, fa, fb,
                    &levelIndex, treePointers,
                    tmColumns, tmRows,
                    upIndecies, downIndecies,
                    columnsIndeciesCount / quantumsCount, quantumsCount);
        } else {
            if (levelIndex == 0) {
                levelIndex++;
            }
            bool next = prepareLevel(levelIndex, treePointers,
                    tmColumns, tmRows,
                    upIndecies, downIndecies,
                    columnsIndeciesCount / quantumsCount, quantumsCount);
            if (next == false) {
                finish = nextBranch(&levelIndex, treePointers,
                        tmColumns, tmRows);
            } else {
                levelIndex++;
            }
        }
    }
}

extern "C" __global__ void ExecuteTM(math::Matrix* transferMatrix,
        TreePointer*** treePointers1,
        intt** tmColumns1,
        intt** tmRows1, char** tmRowsBits1,
        intt* upIndecies, intt* downIndecies,
        intt* columnsIndeciesCountPtr,
        intt* quantumsCountPtr, intt* M2Ptr, intt* width) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    uintt index = threadIndexX + *width * threadIndexY;
    TreePointer** treePointers = treePointers1[index];

    intt* tmColumns = tmColumns1[index];
    intt* tmRows = tmRows1[index];
    char* tmRowsBits = tmRowsBits1[index];
    uintt endColumn = threadIndexX;
    uintt endRow = threadIndexY;
    intt M2 = *M2Ptr;
    uintt nvalues = *quantumsCountPtr * *quantumsCountPtr;


    for (uintt fb = 0; fb < M2; ++fb) {
        tmRowsBits[fb] = 0;
    }

    for (uintt fb = 0; fb < M2 / 2; ++fb) {
        tmRows[fb] = 0;
        tmColumns[fb] = 0;
    }

    for (uintt fb = 0; fb < endRow; ++fb) {
        increment(tmRowsBits, M2, 2, false);
    }
    for (uintt fa = 0; fa < endColumn; ++fa) {
        increment(tmColumns, M2 / 2, nvalues, false);
    }
    convertBitsToIndex(tmRows, tmRowsBits, M2 / 2);
    uintt fa = threadIndexX;
    uintt fb = threadIndexY;
    uintt columnsIndeciesCount = *columnsIndeciesCountPtr;
    intt quantumsCount = *quantumsCountPtr;
    bool finish = false;
    intt levelIndex = 0;

    while (finish == false) {
        if (levelIndex == 0 && treePointers[levelIndex]->index == 0) {
            finish = prepareFirstLevel(levelIndex, treePointers,
                    tmColumns, tmRows,
                    upIndecies, downIndecies,
                    columnsIndeciesCount / quantumsCount, quantumsCount);
            levelIndex++;
        } else if (levelIndex == M2 - 1) {
            finish = calculate(transferMatrix, fa, fb,
                    &levelIndex, treePointers,
                    tmColumns, tmRows,
                    upIndecies, downIndecies,
                    columnsIndeciesCount / quantumsCount, quantumsCount);
        } else {
            if (levelIndex == 0) {
                levelIndex++;
            }
            bool next = prepareLevel(levelIndex, treePointers,
                    tmColumns, tmRows,
                    upIndecies, downIndecies,
                    columnsIndeciesCount / quantumsCount, quantumsCount);
            if (next == false) {
                finish = nextBranch(&levelIndex, treePointers,
                        tmColumns, tmRows);
            } else {
                levelIndex++;
            }
        }
    }
}