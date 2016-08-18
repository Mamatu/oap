
#include "DeviceTreePointerCreator.h"
#include "DeviceMatrixModules.h"
#include "CudaUtils.h"

DeviceTreePointerCreator::DeviceTreePointerCreator() {
}

DeviceTreePointerCreator::~DeviceTreePointerCreator() {
}

TreePointer* DeviceTreePointerCreator::create(intt levelIndex,
        math::Matrix* matrix1,
        math::Matrix* matrix2) {
    TreePointer* treePointer = (TreePointer*) CudaUtils::AllocDeviceMem(sizeof (TreePointer));
    intt zero = 0;
    CudaUtils::CopyHostToDevice(&treePointer->count, &zero, sizeof (zero));
    CudaUtils::CopyHostToDevice(&treePointer->index, &zero, sizeof (zero));
    intt realCount = 0;

    if (levelIndex % 2 != 0) {
        CudaUtils::CopyHostToDevice(&treePointer->matrix, &matrix1, sizeof (matrix1));
        CudaUtils::CopyDeviceToDevice(&treePointer->realCount, &matrix1->rows,
                sizeof (matrix1->rows));
        TreePointerType type = TYPE_COLUMN;
        CudaUtils::CopyHostToDevice(&treePointer->type, &type, sizeof (type));
        realCount = CudaUtils::GetRows(matrix1);
    } else {
        CudaUtils::CopyHostToDevice(&treePointer->matrix, &matrix2,
                sizeof (matrix2));
        CudaUtils::CopyDeviceToDevice(&treePointer->realCount, &matrix2->columns,
                sizeof (matrix2->columns));
        TreePointerType type = TYPE_ROW;
        CudaUtils::CopyHostToDevice(&treePointer->type, &type, sizeof (type));
        realCount = CudaUtils::GetColumns(matrix2);
    }
    intt* nodeValue = (intt*) CudaUtils::AllocDeviceMem(realCount * sizeof (intt));
    floatt* revalues = (floatt*) CudaUtils::AllocDeviceMem(realCount * sizeof (floatt));
    floatt* imvalues = (floatt*) CudaUtils::AllocDeviceMem(realCount * sizeof (floatt));
    CUdeviceptr ptr1 = reinterpret_cast<CUdeviceptr> (nodeValue);
    CUdeviceptr ptr2 = reinterpret_cast<CUdeviceptr> (revalues);
    CUdeviceptr ptr3 = reinterpret_cast<CUdeviceptr> (revalues);
    cuMemsetD32(ptr1, 0, realCount * sizeof (intt));
    cuMemsetD32(ptr2, 0, realCount * sizeof (floatt));
    cuMemsetD32(ptr3, 0, realCount * sizeof (floatt));
    CudaUtils::CopyHostToDevice(&treePointer->nodeValue, &nodeValue, sizeof (intt*));
    CudaUtils::CopyHostToDevice(&treePointer->reValues, &revalues, sizeof (floatt*));
    CudaUtils::CopyHostToDevice(&treePointer->imValues, &imvalues, sizeof (floatt*));
    return treePointer;
}

void DeviceTreePointerCreator::destroy(TreePointer* treePointer) {
    intt* nodeValue = NULL;
    floatt* revalues = NULL;
    floatt* imvalues = NULL;
    CudaUtils::CopyDeviceToHost(&nodeValue, &treePointer->nodeValue, sizeof (intt*));
    CudaUtils::CopyDeviceToHost(&revalues, &treePointer->reValues, sizeof (floatt*));
    CudaUtils::CopyDeviceToHost(&imvalues, &treePointer->imValues, sizeof (floatt*));
    CudaUtils::FreeDeviceMem(nodeValue);
    CudaUtils::FreeDeviceMem(revalues);
    CudaUtils::FreeDeviceMem(imvalues);
    CudaUtils::FreeDeviceMem(treePointer);
}

