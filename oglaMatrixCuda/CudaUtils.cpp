#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <map>

namespace CudaUtils {

    CUdeviceptr GetReValuesAddress(const math::Matrix* matrix) {
        return reinterpret_cast<CUdeviceptr> (&matrix->reValues);
    }

    CUdeviceptr GetImValuesAddress(const math::Matrix* matrix) {
        return reinterpret_cast<CUdeviceptr> (&matrix->imValues);
    }

    CUdeviceptr GetColumnsAddress(const math::Matrix* matrix) {
        return reinterpret_cast<CUdeviceptr> (&matrix->columns);
    }

    CUdeviceptr GetRowsAddress(const math::Matrix* matrix) {
        return reinterpret_cast<CUdeviceptr> (&matrix->rows);
    }

    floatt* GetReValues(const math::Matrix* matrix) {
        floatt* reValues = NULL;
        cuMemcpyDtoH(&reValues, GetReValuesAddress(matrix), sizeof (floatt*));
        return reValues;
    }

    floatt* GetImValues(const math::Matrix* matrix) {
        floatt* imValues = NULL;
        cuMemcpyDtoH(&imValues, GetImValuesAddress(matrix), sizeof (floatt*));
        return imValues;
    }

    uintt GetDeviceColumns(const math::Matrix* matrix) {
        uintt columns = 0;
        cuMemcpyDtoH(&columns, GetColumnsAddress(matrix), sizeof (uintt));
        return columns;
    }

    uintt GetDeviceRows(const math::Matrix* matrix) {
        uintt rows = 0;
        cuMemcpyDtoH(&rows, GetRowsAddress(matrix), sizeof (uintt));
        return rows;
    }

    CUdeviceptr GetReValuesAddress(CUdeviceptr matrixptr) {
        math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
        return reinterpret_cast<CUdeviceptr> (&matrix->reValues);
    }

    CUdeviceptr GetImValuesAddress(CUdeviceptr matrixptr) {
        math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
        return reinterpret_cast<CUdeviceptr> (&matrix->imValues);
    }

    CUdeviceptr GetColumnsAddress(CUdeviceptr matrixptr) {
        math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
        return reinterpret_cast<CUdeviceptr> (&matrix->columns);
    }

    CUdeviceptr GetRowsAddress(CUdeviceptr matrixptr) {
        math::Matrix* matrix = reinterpret_cast<math::Matrix*> (matrixptr);
        return reinterpret_cast<CUdeviceptr> (&matrix->rows);
    }

    floatt* GetReValues(CUdeviceptr matrix) {
        floatt* reValues = NULL;
        cuMemcpyDtoH(&reValues, GetReValuesAddress(matrix), sizeof (floatt*));
        return reValues;
    }

    floatt* GetImValues(CUdeviceptr matrix) {
        floatt* imValues = NULL;
        cuMemcpyDtoH(&imValues, GetImValuesAddress(matrix), sizeof (floatt*));
        return imValues;
    }

    intt GetDeviceColumns(CUdeviceptr matrix) {
        intt columns = 0;
        cuMemcpyDtoH(&columns, GetColumnsAddress(matrix), sizeof (int));
        return columns;
    }

    intt GetDeviceRows(CUdeviceptr matrix) {
        intt rows = 0;
        cuMemcpyDtoH(&rows, GetRowsAddress(matrix), sizeof (int));
        return rows;
    }

    CUdeviceptr AllocMatrix() {
        CUdeviceptr devicePtrMatrix = 0;
        printCuError(cuMemAlloc(&devicePtrMatrix, sizeof (math::Matrix)));
        return devicePtrMatrix;
    }

    CUdeviceptr AllocMatrix(bool allocRe, bool allocIm, intt columns,
            intt rows, floatt revalue, floatt imvalue) {
        CUdeviceptr matrix = AllocMatrix();
        CUdeviceptr matrixRe = 0;
        CUdeviceptr matrixIm = 0;
        if (allocRe) {
            matrixRe = AllocReMatrix(matrix, columns, rows, revalue);
        } else {
            matrixRe = SetReMatrixToNull(matrix);
        }
        if (allocIm) {
            matrixIm = AllocImMatrix(matrix, columns, rows, imvalue);
        } else {
            matrixIm = SetImMatrixToNull(matrix);
        }
        SetVariables(matrix, columns, rows);
        return matrix;
    }

    CUdeviceptr AllocReMatrix(CUdeviceptr devicePtrMatrix, intt columns, intt rows, floatt value) {
        CUdeviceptr devicePtrReValues = 0;
        printCuError(cuMemAlloc(&devicePtrReValues, columns * rows * sizeof (floatt)));
        printCuError(cuMemcpyHtoD(GetReValuesAddress(devicePtrMatrix), &devicePtrReValues, sizeof (CUdeviceptr)));
        unsigned int dvalue = *reinterpret_cast<unsigned int*> (&value);
        cuMemsetD32(devicePtrReValues, dvalue, columns * rows);
        return devicePtrReValues;
    }

    CUdeviceptr AllocImMatrix(CUdeviceptr devicePtrMatrix, intt columns, intt rows, floatt value) {
        CUdeviceptr devicePtrImValues = 0;
        printCuError(cuMemAlloc(&devicePtrImValues, columns * rows * sizeof (floatt)));
        printCuError(cuMemcpyHtoD(GetImValuesAddress(devicePtrMatrix), &devicePtrImValues, sizeof (CUdeviceptr)));
        unsigned int dvalue = *reinterpret_cast<unsigned int*> (&value);
        cuMemsetD32(devicePtrImValues, dvalue, columns * rows);
        return devicePtrImValues;
    }

    CUdeviceptr SetReMatrixToNull(CUdeviceptr devicePtrMatrix) {
        CUdeviceptr buffer = 0;
        printCuError(cuMemcpyHtoD(GetReValuesAddress(devicePtrMatrix), &buffer, sizeof (CUdeviceptr)));
        return 0;
    }

    CUdeviceptr SetImMatrixToNull(CUdeviceptr devicePtrMatrix) {
        CUdeviceptr buffer = 0;
        printCuError(cuMemcpyHtoD(GetImValuesAddress(devicePtrMatrix), &buffer, sizeof (CUdeviceptr)));
        return 0;
    }

    void SetVariables(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows) {
        printCuError(cuMemcpyHtoD(GetColumnsAddress(devicePtrMatrix), &columns, sizeof (intt)));
        printCuError(cuMemcpyHtoD(GetRowsAddress(devicePtrMatrix), &rows, sizeof (intt)));
    }

    void* NewDevice(intt size) {
        CUdeviceptr devicePtr;
        cuMemAlloc(&devicePtr, size);
        cuMemsetD32(devicePtr, 0, size);
        return reinterpret_cast<void*> (devicePtr);
    }

    void* NewDevice(intt size, const void* src) {
        static unsigned int count = 0;
        void* devPtr = NewDevice(size);
        CopyHostToDevice(devPtr, src, size);
        fprintf(stderr, "count = %u \n", count++);
        return devPtr;
    }

    void DeleteDevice(void* devicePtr) {
        if (devicePtr) {
            CUdeviceptr deviecPtr = reinterpret_cast<CUdeviceptr> (devicePtr);
            DeleteDevice(deviecPtr);
        }
    }

    void DeleteDevice(CUdeviceptr ptr) {
        if (ptr != 0) {
            cuMemFree(ptr);
        }
    }

    void CopyHostToDevice(void* dst, const void* src, intt size) {
        CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr> (dst);
        cuMemcpyHtoD(dstPtr, src, size);
    }

    void CopyDeviceToHost(void* dst, const void* src, intt size) {
        CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr> (src);
        cuMemcpyDtoH(dst, srcPtr, size);
    }

    void CopyDeviceToDevice(void* dst, const void* src, intt size) {
        CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr> (dst);
        CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr> (src);
        cuMemcpyDtoD(dstPtr, srcPtr, size);
    }
}