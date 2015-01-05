#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <map>

void HDMatrixCopier::copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) {
    cuda::CopyHostMatrixToDeviceMatrix(dst, src);
}

void HDMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = CudaUtils::GetColumns(dst) * CudaUtils::GetRows(dst);
    uintt length2 = src->columns * src->rows;
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr dstRePtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetReValues(dst));
    if (dstRePtr != 0 && src->reValues != NULL) {
        cuMemcpyHtoD(dstRePtr, src->reValues, length1 * sizeof (floatt));
    }
}

void HDMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) {
    uintt length1 = CudaUtils::GetColumns(dst) * CudaUtils::GetRows(dst);
    uintt length2 = src->columns * src->rows;
    length1 = length1 < length2 ? length1 : length2;
    CUdeviceptr dstImPtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetImValues(dst));
    if (dstImPtr != 0 && src->imValues != NULL) {
        cuMemcpyHtoD(dstImPtr, src->imValues, length1 * sizeof (floatt));
    }
}

void HDMatrixCopier::copy(floatt* dst, const floatt* src, uintt length) {
    debugFuncBegin();
    CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr> (dst);
    cuMemcpyHtoD(devicePtr, src, length * sizeof (floatt));
    debugFuncEnd();
}

void HDMatrixCopier::setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        CUdeviceptr matrixRePtr1 = matrixRePtr + column + matrix->columns * fa;
        cuMemcpyHtoD(matrixRePtr1, vector + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void HDMatrixCopier::setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
    matrixRePtr = matrixRePtr + matrix->columns * row;
    cuMemcpyHtoD(matrixRePtr, vector, length * sizeof (floatt));
    debugFuncEnd();
}

void HDMatrixCopier::setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
    for (uintt fa = 0; fa < length; fa++) {
        CUdeviceptr matrixImPtr1 = matrixImPtr + column + matrix->columns * fa;
        cuMemcpyHtoD(matrixImPtr1, vector + fa, sizeof (floatt));
    }
    debugFuncEnd();
}

void HDMatrixCopier::setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length) {
    debugFuncBegin();
    CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
    matrixImPtr = matrixImPtr + matrix->columns * row;
    cuMemcpyHtoD(matrixImPtr, vector, length * sizeof (floatt));
    debugFuncEnd();
}

void HDMatrixCopier::getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    for (uintt fa = 0; fa < length; fa++) {
        cuMemcpyHtoD(vectorPtr + fa, &matrix->reValues[column + matrix->columns * fa],
                sizeof (floatt));
    }
    debugFuncEnd();
}

void HDMatrixCopier::getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    cuMemcpyHtoD(vectorPtr, &matrix->reValues[matrix->columns * row],
            length * sizeof (floatt));
    debugFuncEnd();
}

void HDMatrixCopier::getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    for (uintt fa = 0; fa < length; fa++) {
        cuMemcpyHtoD(vectorPtr + fa, &matrix->imValues[column + matrix->columns * fa],
                sizeof (floatt));
    }
    debugFuncEnd();
}

void HDMatrixCopier::getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row) {
    debugFuncBegin();
    CUdeviceptr vectorPtr = reinterpret_cast<CUdeviceptr> (vector);
    cuMemcpyHtoD(vectorPtr, &matrix->imValues[matrix->columns * row],
            length * sizeof (floatt));
    debugFuncEnd();
}

