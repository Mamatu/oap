/* 
 * File:   DeviceTreePointerCreator.cpp
 * Author: mmatula
 * 
 * Created on June 15, 2014, 10:36 AM
 */

#include "DeviceTreePointerCreator.h"
#include "DeviceMatrixModules.h"

DeviceTreePointerCreator::DeviceTreePointerCreator() {
}

DeviceTreePointerCreator::~DeviceTreePointerCreator() {
}

TreePointer* DeviceTreePointerCreator::create(intt levelIndex,
        math::Matrix* matrix1,
        math::Matrix* matrix2) {
    TreePointer* treePointer = (TreePointer*) device::NewDevice(sizeof (TreePointer));
    DeviceMatrixUtils dmu;
    intt zero = 0;
    device::CopyHostToDevice(&treePointer->count, &zero, sizeof (zero));
    device::CopyHostToDevice(&treePointer->index, &zero, sizeof (zero));
    intt realCount = 0;

    if (levelIndex % 2 != 0) {
        device::CopyHostToDevice(&treePointer->matrix, &matrix1, sizeof (matrix1));
        device::CopyDeviceToDevice(&treePointer->realCount, &matrix1->rows,
                sizeof (matrix1->rows));
        TreePointerType type = TYPE_COLUMN;
        device::CopyHostToDevice(&treePointer->type, &type, sizeof (type));
        realCount = dmu.getRows(matrix1);
    } else {
        device::CopyHostToDevice(&treePointer->matrix, &matrix2,
                sizeof (matrix2));
        device::CopyDeviceToDevice(&treePointer->realCount, &matrix2->columns,
                sizeof (matrix2->columns));
        TreePointerType type = TYPE_ROW;
        device::CopyHostToDevice(&treePointer->type, &type, sizeof (type));
        realCount = dmu.getColumns(matrix2);
    }
    intt* nodeValue = (intt*) device::NewDevice(realCount * sizeof (intt));
    floatt* revalues = (floatt*) device::NewDevice(realCount * sizeof (floatt));
    floatt* imvalues = (floatt*) device::NewDevice(realCount * sizeof (floatt));
    CUdeviceptr ptr1 = reinterpret_cast<CUdeviceptr> (nodeValue);
    CUdeviceptr ptr2 = reinterpret_cast<CUdeviceptr> (revalues);
    CUdeviceptr ptr3 = reinterpret_cast<CUdeviceptr> (revalues);
    cuMemsetD32(ptr1, 0, realCount * sizeof (intt));
    cuMemsetD32(ptr2, 0, realCount * sizeof (floatt));
    cuMemsetD32(ptr3, 0, realCount * sizeof (floatt));
    device::CopyHostToDevice(&treePointer->nodeValue, &nodeValue, sizeof (intt*));
    device::CopyHostToDevice(&treePointer->reValues, &revalues, sizeof (floatt*));
    device::CopyHostToDevice(&treePointer->imValues, &imvalues, sizeof (floatt*));
    return treePointer;
}

void DeviceTreePointerCreator::destroy(TreePointer* treePointer) {
    intt* nodeValue = NULL;
    floatt* revalues = NULL;
    floatt* imvalues = NULL;
    device::CopyDeviceToHost(&nodeValue, &treePointer->nodeValue, sizeof (intt*));
    device::CopyDeviceToHost(&revalues, &treePointer->reValues, sizeof (floatt*));
    device::CopyDeviceToHost(&imvalues, &treePointer->imValues, sizeof (floatt*));
    device::DeleteDevice(nodeValue);
    device::DeleteDevice(revalues);
    device::DeleteDevice(imvalues);
    device::DeleteDevice(treePointer);
}

