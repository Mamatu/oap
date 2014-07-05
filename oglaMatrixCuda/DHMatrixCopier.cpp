#include "DeviceMatrixModules.h"

void DHMatrixCopier::copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = dst->columns * dst->rows;
    uintt length2 = getDeviceColumns(src) * getDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr srcRePtr = reinterpret_cast<CUdeviceptr>(getReValues(src));
    CUdeviceptr srcImPtr = reinterpret_cast<CUdeviceptr>(getImValues(src));
    if (srcRePtr != 0 && dst->reValues != NULL) {
        cuMemcpyDtoH(dst->reValues, srcRePtr, length1 * sizeof (floatt));
    }
    if (srcImPtr != 0 && dst->imValues != NULL) {
        cuMemcpyDtoH(dst->imValues, srcImPtr, length1 * sizeof (floatt));
    }
}

void DHMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = dst->columns * dst->rows;
    uintt length2 = getDeviceColumns(src) * getDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr srcRePtr = reinterpret_cast<CUdeviceptr>(getReValues(src));
    if (srcRePtr != 0 && dst->reValues != NULL) {
        cuMemcpyDtoH(dst->reValues, srcRePtr, length1 * sizeof (floatt));
    }
}

void DHMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = dst->columns * dst->rows;
    uintt length2 = getDeviceColumns(src) * getDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr srcImPtr = reinterpret_cast<CUdeviceptr>(getImValues(src));
    if (srcImPtr != 0 && dst->imValues != NULL) {
        cuMemcpyDtoH(dst->imValues, srcImPtr, length1 * sizeof (floatt));
    }
}

void DHMatrixCopier::copy(floatt* dst, const floatt* src, intt length) {
    debugFuncBegin();
    CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr> (src);
    cuMemcpyDtoH(dst, srcPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::setReVector(math::Matrix* matrix, intt column,
        floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    for (intt fa = 0; fa < length; fa++) {
        cuMemcpyDtoH(&matrix->reValues[column + matrix->columns * fa],
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::setTransposeReVector(math::Matrix* matrix, intt row,
        floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    cuMemcpyDtoH(&matrix->reValues[matrix->columns * row],
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::setImVector(math::Matrix* matrix, intt column,
        floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    for (intt fa = 0; fa < length; fa++) {
        cuMemcpyDtoH(&matrix->imValues[column + matrix->columns * fa],
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::setTransposeImVector(math::Matrix* matrix, intt row,
        floatt* vector, intt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    cuMemcpyDtoH(&matrix->reValues[matrix->columns * row],
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::getReVector(floatt* vector, intt length,
        math::Matrix* matrix, intt column) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = getReValuesAddress(matrix);
    intt columns = getDeviceColumns(matrix);
    for (intt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixRePtr1 = matrixRePtr + fa * columns + column;
        cuMemcpyDtoH(vector + fa, matrixRePtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::getTransposeReVector(floatt* vector, intt length,
        math::Matrix* matrix, intt row) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = this->getReValuesAddress(matrix);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyDtoH(vector, matrixRePtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::getImVector(floatt* vector, intt length,
        math::Matrix* matrix, intt column) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = getImValuesAddress(matrix);
    intt columns = getDeviceColumns(matrix);
    for (intt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixRePtr1 = matrixRePtr + fa * columns + column;
        cuMemcpyDtoH(vector + fa, matrixRePtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::getTransposeImVector(floatt* vector, intt length,
        math::Matrix* matrix, intt row) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = this->getImValuesAddress(matrix);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyDtoH(vector, matrixRePtr, length * sizeof (floatt));
    debugFuncEnd();
}
