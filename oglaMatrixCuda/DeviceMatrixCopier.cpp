#include "DeviceMatrixModules.h"

void DeviceMatrixCopier::copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) {
    debugFuncBegin();
    intt length1 = getDeviceColumns(dst) * getDeviceRows(dst);
    intt length2 = getDeviceColumns(src) * getDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    cuMemcpyDtoD(getReValuesAddress(dst), getReValuesAddress(src), length1 * sizeof (floatt));
    cuMemcpyDtoD(getImValuesAddress(dst), getImValuesAddress(src), length1 * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) {
    debugFuncBegin();
    intt length1 = getDeviceColumns(dst) * getDeviceRows(dst);
    intt length2 = getDeviceColumns(src) * getDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    cuMemcpyDtoD(getReValuesAddress(dst), getReValuesAddress(src), length1 * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) {
    debugFuncBegin();
    intt length1 = getDeviceColumns(dst) * getDeviceRows(dst);
    intt length2 = getDeviceColumns(src) * getDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    cuMemcpyDtoD(getImValuesAddress(dst), getImValuesAddress(src), length1 * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::copy(floatt* dst, const floatt* src, intt length) {
    debugFuncBegin();
    CUdeviceptr dstptr = reinterpret_cast<CUdeviceptr> (dst);
    CUdeviceptr srcptr = reinterpret_cast<CUdeviceptr> (src);
    cuMemcpyDtoD(dstptr, srcptr, sizeof (floatt) * length);
    debugFuncEnd();
}

void DeviceMatrixCopier::setReVector(math::Matrix* matrix, intt column, floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixRePtr = getReValuesAddress(matrix);
    for (intt fa = 0; fa < length; fa++) {
        cuMemcpyDtoD(matrixRePtr + column + matrix->columns * fa,
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::setTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixRePtr = getReValuesAddress(matrix);
    cuMemcpyDtoD(matrixRePtr + matrix->columns * row,
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::setImVector(math::Matrix* matrix, intt column, floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixImPtr = getImValuesAddress(matrix);
    for (intt fa = 0; fa < length; fa++) {
        cuMemcpyDtoD(matrixImPtr + column + matrix->columns * fa,
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::setTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixImPtr = getImValuesAddress(matrix);
    cuMemcpyDtoD(matrixImPtr + matrix->columns * row,
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::getReVector(floatt* vector, intt length, math::Matrix* matrix, intt column) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = getReValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    intt columns = getDeviceColumns(matrix);
    for (intt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixRePtr1 = matrixRePtr + fa * columns + column;
        cuMemcpyDtoD(vectorPtr + fa, matrixRePtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::getTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = this->getReValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyDtoD(vectorPtr, matrixRePtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::getImVector(floatt* vector, intt length, math::Matrix* matrix, intt column) {
    debugFuncBegin();
    CUdeviceptr matrixImPtr = getImValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    intt columns = getDeviceColumns(matrix);
    for (intt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixImPtr1 = matrixImPtr + fa * columns + column;
        cuMemcpyDtoD(vectorPtr + fa, matrixImPtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::getTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row) {
    debugFuncBegin();
    CUdeviceptr matrixImPtr = this->getImValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    matrixImPtr = matrixImPtr + matrix->columns * row;
    cuMemcpyDtoD(vectorPtr, matrixImPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows) {

}

void DeviceMatrixCopier::getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column) {

}

void HDMatrixCopier::setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows) {

}

void HDMatrixCopier::getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column) {

}

void DHMatrixCopier::setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows) {

}

void DHMatrixCopier::getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column) {

}
