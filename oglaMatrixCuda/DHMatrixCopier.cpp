#include "DeviceMatrixModules.h"

void DHMatrixCopier::copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) {
    cuda::CopyDeviceMatrixToHostMatrix(dst, src);
}

void DHMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = dst->columns * dst->rows;
    uintt length2 = CudaUtils::GetColumns(src) * CudaUtils::GetRows(src);
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr srcRePtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetReValues(src));
    if (srcRePtr != 0 && dst->reValues != NULL) {
        cuMemcpyDtoH(dst->reValues, srcRePtr, length1 * sizeof (floatt));
    }
}

void DHMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = dst->columns * dst->rows;
    uintt length2 = CudaUtils::GetColumns(src) * CudaUtils::GetRows(src);
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr srcImPtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetImValues(src));
    if (srcImPtr != 0 && dst->imValues != NULL) {
        cuMemcpyDtoH(dst->imValues, srcImPtr, length1 * sizeof (floatt));
    }
}

void DHMatrixCopier::copy(floatt* dst, const floatt* src, uintt length) {
    debugFuncBegin();
    CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr> (src);
    cuMemcpyDtoH(dst, srcPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::setReVector(math::Matrix* matrix, uintt column,
        floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    for (uintt fa = 0; fa < length; fa++) {
        cuMemcpyDtoH(&matrix->reValues[column + matrix->columns * fa],
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::setTransposeReVector(math::Matrix* matrix, uintt row,
        floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    cuMemcpyDtoH(&matrix->reValues[matrix->columns * row],
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::setImVector(math::Matrix* matrix, uintt column,
        floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    for (uintt fa = 0; fa < length; fa++) {
        cuMemcpyDtoH(&matrix->imValues[column + matrix->columns * fa],
                vectorPtr + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::setTransposeImVector(math::Matrix* matrix, uintt row,
        floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    cuMemcpyDtoH(&matrix->reValues[matrix->columns * row],
            vectorPtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::getReVector(floatt* vector, uintt length,
        math::Matrix* matrix, uintt column) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    uintt columns = CudaUtils::GetColumns(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixRePtr1 = matrixRePtr + fa * columns + column;
        cuMemcpyDtoH(vector + fa, matrixRePtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::getTransposeReVector(floatt* vector, uintt length,
        math::Matrix* matrix, uintt row) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyDtoH(vector, matrixRePtr, length * sizeof (floatt));
    debugFuncEnd();
}

void DHMatrixCopier::getImVector(floatt* vector, uintt length,
        math::Matrix* matrix, uintt column) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetImValuesAddress(matrix);
    uintt columns = CudaUtils::GetColumns(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        const CUdeviceptr matrixRePtr1 = matrixRePtr + fa * columns + column;
        cuMemcpyDtoH(vector + fa, matrixRePtr1, sizeof (floatt));
    }
    debugFuncEnd();
}

void DHMatrixCopier::getTransposeImVector(floatt* vector, uintt length,
        math::Matrix* matrix, uintt row) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetImValuesAddress(matrix);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyDtoH(vector, matrixRePtr, length * sizeof (floatt));
    debugFuncEnd();
}
