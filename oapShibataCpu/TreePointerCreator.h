
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

