
#ifndef DEVICETREEPOINTERCREATOR_H
#define	DEVICETREEPOINTERCREATOR_H

#include "TreePointer.h"
#include "TreePointerCreator.h"

class DeviceTreePointerCreator : public TreePointerCreator {
public:
    DeviceTreePointerCreator();
    virtual ~DeviceTreePointerCreator();
    TreePointer* create(intt levelIndex,
            math::Matrix* matrix1,
            math::Matrix* matrix2);

    void destroy(TreePointer* treePointer);
};

#endif	/* DEVICETREEPOINTERCREATOR_H */

