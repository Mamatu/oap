#include "DeviceMatrixModules.h"

void DeviceMatrixCopier::copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) {
    debugFuncBegin();
    uintt length1 = CudaUtils::GetDeviceColumns(dst) * CudaUtils::GetDeviceRows(dst);
    uintt length2 = CudaUtils::GetDeviceColumns(src) * CudaUtils::GetDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    cuMemcpyDtoD(CudaUtils::GetReValuesAddress(dst), CudaUtils::GetReValuesAddress(src), length1 * sizeof (floatt));
    cuMemcpyDtoD(CudaUtils::GetImValuesAddress(dst), CudaUtils::GetImValuesAddress(src), length1 * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) {
    debugFuncBegin();
    uintt length1 = CudaUtils::GetDeviceColumns(dst) * CudaUtils::GetDeviceRows(dst);
    uintt length2 = CudaUtils::GetDeviceColumns(src) * CudaUtils::GetDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    cuMemcpyDtoD(CudaUtils::GetReValuesAddress(dst), CudaUtils::GetReValuesAddress(src), length1 * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) {
    debugFuncBegin();
    uintt length1 = CudaUtils::GetDeviceColumns(dst) * CudaUtils::GetDeviceRows(dst);
    uintt length2 = CudaUtils::GetDeviceColumns(src) * CudaUtils::GetDeviceRows(src);
    length1 = length1 < length2 ? length1 : length2;
    cuMemcpyDtoD(CudaUtils::GetImValuesAddress(dst), CudaUtils::GetImValuesAddress(src), length1 * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::copy(floatt* dst, const floatt* src, uintt length) {
    debugFuncBegin();
    CUdeviceptr dstptr = reinterpret_cast<CUdeviceptr> (dst);
    CUdeviceptr srcptr = reinterpret_cast<CUdeviceptr> (src);
    cuMemcpyDtoD(dstptr, srcptr, sizeof (floatt) * length);
    debugFuncEnd();
}

void DeviceMatrixCopier::setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        cuMemcpyDtoD(matrixRePtr + column + matrix->columns * fa,
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    cuMemcpyDtoD(matrixRePtr + matrix->columns * row,
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        cuMemcpyDtoD(matrixImPtr + column + matrix->columns * fa,
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
    cuMemcpyDtoD(matrixImPtr + matrix->columns * row,
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    uintt columns = CudaUtils::GetDeviceColumns(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixRePtr1 = matrixRePtr + fa * columns + column;
        cuMemcpyDtoD(vectorPtr + fa, matrixRePtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyDtoD(vectorPtr, matrixRePtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column) {
    debugFuncBegin();
    CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    uintt columns = CudaUtils::GetDeviceColumns(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixImPtr1 = matrixImPtr + fa * columns + column;
        cuMemcpyDtoD(vectorPtr + fa, matrixImPtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DeviceMatrixCopier::getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row) {
    debugFuncBegin();
    CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    matrixImPtr = matrixImPtr + matrix->columns * row;
    cuMemcpyDtoD(vectorPtr, matrixImPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DeviceMatrixCopier::setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows) {

}

void DeviceMatrixCopier::getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column) {

}

void HDMatrixCopier::setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows) {

}

void HDMatrixCopier::getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column) {

}

void DHMatrixCopier::setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows) {

}

void DHMatrixCopier::getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column) {

}
