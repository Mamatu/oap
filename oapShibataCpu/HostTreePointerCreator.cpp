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




#include "TreePointerCreator.h"
#include <string.h>

TreePointerCreator::TreePointerCreator() {
}

TreePointerCreator::~TreePointerCreator() {
}

HostTreePointerCreator::HostTreePointerCreator() : TreePointerCreator() {
}

HostTreePointerCreator::~HostTreePointerCreator() {
}

TreePointer* HostTreePointerCreator::create(intt levelIndex,
        math::Matrix* matrix1,
        math::Matrix* matrix2) {
    TreePointer* treePointer = new TreePointer();
    treePointer->realCount = 0;
    treePointer->count = 0;
    treePointer->index = 0;
    if (levelIndex % 2 != 0) {
        treePointer->realCount = matrix1->rows;
        treePointer->matrix = matrix1;
        treePointer->type = TYPE_COLUMN;
    } else {
        treePointer->matrix = matrix2;
        treePointer->realCount = matrix2->columns;
        treePointer->type = TYPE_ROW;
    }
    treePointer->nodeValue = new intt[treePointer->realCount];
    treePointer->reValues = new floatt[treePointer->realCount];
    treePointer->imValues = new floatt[treePointer->realCount];
    memset(treePointer->nodeValue, 0, sizeof (intt) *
            treePointer->realCount);
    memset(treePointer->nodeValue, 0, sizeof (floatt) *
            treePointer->realCount);
    return treePointer;
}

void HostTreePointerCreator::destroy(TreePointer* treePointer) {
    delete[] treePointer->nodeValue;
    delete[] treePointer->reValues;
    delete[] treePointer->imValues;
    delete treePointer;
}
