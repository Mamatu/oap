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




#ifndef TREEPOINTERCREATOR_H
#define	TREEPOINTERCREATOR_H

#include "TreePointer.h"

class TreePointerCreator {
public:
    TreePointerCreator();
    virtual ~TreePointerCreator();
    virtual TreePointer* create(intt levelIndex,
            math::Matrix* matrix1,
            math::Matrix* matrix2) = 0;
    virtual void destroy(TreePointer* treePointer) = 0;
};

class HostTreePointerCreator : public TreePointerCreator {
public:
    HostTreePointerCreator();
    virtual ~HostTreePointerCreator();
    TreePointer* create(intt levelIndex,
            math::Matrix* matrix1,
            math::Matrix* matrix2);
    void destroy(TreePointer* treePointer);
};

#endif	/* TREEPOINTERCREATOR_H */
