#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <map>

CUdeviceptr DeviceUtils::getReValuesAddress(const math::Matrix* matrix) const {
    return reinterpret_cast<CUdeviceptr> (&matrix->reValues);
}

CUdeviceptr DeviceUtils::getImValuesAddress(const math::Matrix* matrix) const {
    return reinterpret_cast<CUdeviceptr> (&matrix->imValues);
}

CUdeviceptr DeviceUtils::getColumnsAddress(const math::Matrix* matrix) const {
    return reinterpret_cast<CUdeviceptr> (&matrix->columns);
}

CUdeviceptr DeviceUtils::getRowsAddress(const math::Matrix* matrix) const {
    return reinterpret_cast<CUdeviceptr> (&matrix->rows);
}

floatt* DeviceUtils::getReValues(const math::Matrix* matrix) const {
    floatt* reValues = NULL;
    cuMemcpyDtoH(&reValues, getReValuesAddress(matrix), sizeof (floatt*));
    return reValues;
}

floatt* DeviceUtils::getImValues(const math::Matrix* matrix) const {
    floatt* imValues = NULL;
    cuMemcpyDtoH(&imValues, getImValuesAddress(matrix), sizeof (floatt*));
    return imValues;
}

uintt DeviceUtils::getDeviceColumns(const math::Matrix* matrix) const {
    uintt columns = 0;
    cuMemcpyDtoH(&columns, getColumnsAddress(matrix), sizeof (uintt));
    return columns;
}

uintt DeviceUtils::getDeviceRows(const math::Matrix* matrix) const {
    uintt rows = 0;
    cuMemcpyDtoH(&rows, getRowsAddress(matrix), sizeof (uintt));
    return rows;
}

CUdeviceptr DeviceUtils::getReValuesAddress(CUdeviceptr matrixptr) const {
    math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
    return reinterpret_cast<CUdeviceptr> (&matrix->reValues);
}

CUdeviceptr DeviceUtils::getImValuesAddress(CUdeviceptr matrixptr) const {
    math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
    return reinterpret_cast<CUdeviceptr> (&matrix->imValues);
}

CUdeviceptr DeviceUtils::getColumnsAddress(CUdeviceptr matrixptr) const {
    math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
    return reinterpret_cast<CUdeviceptr> (&matrix->columns);
}

CUdeviceptr DeviceUtils::getRowsAddress(CUdeviceptr matrixptr) const {
    math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
    return reinterpret_cast<CUdeviceptr> (&matrix->rows);
}

floatt* DeviceUtils::getReValues(CUdeviceptr matrix) const {
    floatt* reValues = NULL;
    cuMemcpyDtoH(&reValues, getReValuesAddress(matrix), sizeof (floatt*));
    return reValues;
}

floatt* DeviceUtils::getImValues(CUdeviceptr matrix) const {
    floatt* imValues = NULL;
    cuMemcpyDtoH(&imValues, getImValuesAddress(matrix), sizeof (floatt*));
    return imValues;
}

intt DeviceUtils::getDeviceColumns(CUdeviceptr matrix) const {
    intt columns = 0;
    cuMemcpyDtoH(&columns, getColumnsAddress(matrix), sizeof (int));
    return columns;
}

intt DeviceUtils::getDeviceRows(CUdeviceptr matrix) const {
    intt rows = 0;
    cuMemcpyDtoH(&rows, getRowsAddress(matrix), sizeof (int));
    return rows;
}

CUdeviceptr DeviceUtils::allocMatrix() {
    debugFuncBegin();
    CUdeviceptr devicePtrMatrix = 0;
    printCuError(cuMemAlloc(&devicePtrMatrix, sizeof (math::Matrix)));
    debugFuncEnd();
    return devicePtrMatrix;
}

CUdeviceptr DeviceUtils::allocMatrix(bool allocRe, bool allocIm, intt columns,
        intt rows, floatt revalue, floatt imvalue) {
    CUdeviceptr matrix = allocMatrix();
    CUdeviceptr matrixRe = 0;
    CUdeviceptr matrixIm = 0;
    if (allocRe) {
        matrixRe = allocReMatrix(matrix, columns, rows, revalue);
    } else {
        matrixRe = setReMatrixToNull(matrix);
    }
    if (allocIm) {
        matrixIm = allocImMatrix(matrix, columns, rows, imvalue);
    } else {
        matrixIm = setImMatrixToNull(matrix);
    }
    setVariables(matrix, columns, rows);
    return matrix;
}

CUdeviceptr DeviceUtils::allocReMatrix(CUdeviceptr devicePtrMatrix, intt columns, intt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr devicePtrReValues = 0;
    printCuError(cuMemAlloc(&devicePtrReValues, columns * rows * sizeof (floatt)));
    printCuError(cuMemcpyHtoD(getReValuesAddress(devicePtrMatrix), &devicePtrReValues, sizeof (CUdeviceptr)));
    unsigned int dvalue = *reinterpret_cast<unsigned int*> (&value);
    cuMemsetD32(devicePtrReValues, dvalue, columns * rows);
    debugFuncEnd();
    return devicePtrReValues;
}

CUdeviceptr DeviceUtils::allocImMatrix(CUdeviceptr devicePtrMatrix, intt columns, intt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr devicePtrImValues = 0;
    printCuError(cuMemAlloc(&devicePtrImValues, columns * rows * sizeof (floatt)));
    printCuError(cuMemcpyHtoD(getImValuesAddress(devicePtrMatrix), &devicePtrImValues, sizeof (CUdeviceptr)));
    unsigned int dvalue = *reinterpret_cast<unsigned int*> (&value);
    cuMemsetD32(devicePtrImValues, dvalue, columns * rows);
    debugFuncEnd();
    return devicePtrImValues;
}

CUdeviceptr DeviceUtils::setReMatrixToNull(CUdeviceptr devicePtrMatrix) {
    debugFuncBegin();
    CUdeviceptr buffer = 0;
    printCuError(cuMemcpyHtoD(getReValuesAddress(devicePtrMatrix), &buffer, sizeof (CUdeviceptr)));
    debugFuncEnd();
    return 0;
}

CUdeviceptr DeviceUtils::setImMatrixToNull(CUdeviceptr devicePtrMatrix) {
    debugFuncBegin();
    CUdeviceptr buffer = 0;
    printCuError(cuMemcpyHtoD(getImValuesAddress(devicePtrMatrix), &buffer, sizeof (CUdeviceptr)));
    debugFuncEnd();
    return 0;
}

void DeviceUtils::setVariables(CUdeviceptr devicePtrMatrix,
        intt columns, intt rows) {
    debugFuncBegin();
    printCuError(cuMemcpyHtoD(getColumnsAddress(devicePtrMatrix), &columns, sizeof (intt)));
    printCuError(cuMemcpyHtoD(getRowsAddress(devicePtrMatrix), &rows, sizeof (intt)));
    debugFuncEnd();
}

