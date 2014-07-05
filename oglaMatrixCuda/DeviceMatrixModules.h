#ifndef OGLA_DEVICE_MATRIX_UTILS_H
#define	OGLA_DEVICE_MATRIX_UTILS_H
#include <stdio.h>
#include "HostMatrixModules.h"
#include "Matrix.h"
#include "ThreadUtils.h"
#include <cuda.h>
#include <map>

class DeviceUtils {
public:
    CUdeviceptr getReValuesAddress(const math::Matrix* matrix) const;
    CUdeviceptr getImValuesAddress(const math::Matrix* matrix) const;
    CUdeviceptr getColumnsAddress(const math::Matrix* matrix) const;
    CUdeviceptr getRowsAddress(const math::Matrix* matrix) const;
    floatt* getReValues(const math::Matrix* matrix) const;
    floatt* getImValues(const math::Matrix* matrix) const;
    uintt getDeviceColumns(const math::Matrix* matrix) const;
    uintt getDeviceRows(const math::Matrix* matrix) const;
    CUdeviceptr getReValuesAddress(CUdeviceptr matrix) const;
    CUdeviceptr getImValuesAddress(CUdeviceptr matrix) const;
    CUdeviceptr getColumnsAddress(CUdeviceptr matrix) const;
    CUdeviceptr getRowsAddress(CUdeviceptr matrix) const;
    floatt* getReValues(CUdeviceptr matrix) const;
    floatt* getImValues(CUdeviceptr matrix) const;
    intt getDeviceColumns(CUdeviceptr matrix) const;
    intt getDeviceRows(CUdeviceptr matrix) const;
    CUdeviceptr allocMatrix();
    CUdeviceptr allocMatrix(bool allocRe, bool allocIm, intt columns,
            intt rows, floatt revalue = 0, floatt imvalue = 0);
    CUdeviceptr allocReMatrix(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows, floatt value);
    CUdeviceptr allocImMatrix(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows, floatt value);
    CUdeviceptr setReMatrixToNull(CUdeviceptr devicePtrMatrix);
    CUdeviceptr setImMatrixToNull(CUdeviceptr devicePtrMatrix);
    void setVariables(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows);
};

class DeviceMatrixAllocator : public MatrixAllocator, public DeviceUtils {
    HostMatrixAllocator hma;
    static synchronization::RecursiveMutex mutex;
    static void lock();
    static void unlock();
    friend class DeviceMatrixUtils;
    friend class DeviceMatrixCopier;
    friend class DeviceMatrixPrinter;
    friend class HDMatrixCopier;
    friend class DHMatrixCopier;
public:
    DeviceMatrixAllocator();
    ~DeviceMatrixAllocator();
    math::Matrix* newReMatrix(intt columns, intt rows, floatt value = 0);
    math::Matrix* newImMatrix(intt columns, intt rows, floatt value = 0);
    math::Matrix* newMatrix(intt columns, intt rows, floatt value = 0);
    bool isMatrix(math::Matrix* matrix);
    math::Matrix* newMatrixFromAsciiFile(const char* path);
    math::Matrix* newMatrixFromBinaryFile(const char* path);
    void deleteMatrix(math::Matrix* matrix);
};

class DeviceMatrixUtils : public MatrixUtils, public DeviceUtils {
    DeviceMatrixAllocator dma;
public:
    DeviceMatrixUtils();
    ~DeviceMatrixUtils();
    void setDiagonalReMatrix(math::Matrix* matrix, floatt value);
    void setDiagonalImMatrix(math::Matrix* matrix, floatt value);
    void setZeroReMatrix(math::Matrix* matrix);
    void setZeroImMatrix(math::Matrix* matrix);
    intt getColumns(const math::Matrix* matrix) const;
    intt getRows(const math::Matrix* matrix) const;
    bool isReMatrix(const math::Matrix* matrix) const;
    bool isImMatrix(const math::Matrix* matrix) const;
    void printInfo(const math::Matrix* matrix) const;
};

class DeviceMatrixCopier : public MatrixCopier, public DeviceUtils {
public:
    DeviceMatrixCopier();
    virtual ~DeviceMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    void copy(floatt* dst, const floatt* src, intt length);
    void setReVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void setImVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void getReVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void getImVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column);
};

/**
 * This class allows to copy from Host to Device.
 */
class HDMatrixCopier : public MatrixCopier, public DeviceUtils {
public:
    HDMatrixCopier();
    virtual ~HDMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    void copy(floatt* dst, const floatt* src, intt length);
    void setReVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void setImVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void getReVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void getImVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column);
};

/**
 * This class allows to copy from Device to Host.
 */
class DHMatrixCopier : public MatrixCopier, public DeviceUtils {
public:
    DHMatrixCopier();
    virtual ~DHMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    void copy(floatt* dst, const floatt* src, intt length);
    void setReVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void setImVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void getReVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void getImVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row);

    void setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column);
};

class DeviceMatrixPrinter : public MatrixPrinter {
    HostMatrixPrinter hmp;
    HostMatrixAllocator hma;
    DHMatrixCopier dhcopier;
public:
    DeviceMatrixPrinter();
    ~DeviceMatrixPrinter();
    void getReMatrixStr(std::string& str, const math::Matrix* matrix);
    void getImMatrixStr(std::string& str, const math::Matrix* matrix);
    void getMatrixStr(std::string& str, const math::Matrix* matrix);
    void printReMatrix(FILE* stream, const math::Matrix* matrix);
    void printReMatrix(const math::Matrix* matrix);
    void printReMatrix(const std::string& text, const math::Matrix* matrix);
    void printImMatrix(FILE* stream, const math::Matrix* matrix);
    void printImMatrix(const math::Matrix* matrix);
    void printImMatrix(const std::string& text, const math::Matrix* matrix);
};

class DeviceMatrixModules : public MatrixModule {
    DeviceMatrixAllocator dma;
    DeviceMatrixCopier dmc;
    DeviceMatrixUtils dmu;
    DeviceMatrixPrinter dmp;
    HDMatrixCopier hdmc;
    DHMatrixCopier dhmc;
    static DeviceMatrixModules deviceMatrixMoules;
protected:
    DeviceMatrixModules();
    virtual ~DeviceMatrixModules();
public:
    math::Matrix* newDeviceMatrix(math::Matrix* hostMatrix);
    static DeviceMatrixModules& getInstance();
    virtual MatrixAllocator* getMatrixAllocator();
    virtual MatrixCopier* getMatrixCopier();
    virtual HDMatrixCopier* getHDCopier();
    virtual DHMatrixCopier* getDHCopier();
    virtual MatrixUtils* getMatrixUtils();
    virtual MatrixPrinter* getMatrixPrinter();
};

namespace device {
    math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix);
    math::Matrix* NewDeviceMatrix(math::Matrix* hostMatrix);
    template<typename T>T* NewDeviceValue(T v = 0);
    template<typename T>void DeleteDeviceValue(T* valuePtr);
    void DeleteDeviceMatrix(math::Matrix* deviceMatrix);
    void CopyDeviceMatrixToHostMatrix(math::Matrix* hostMatrix, math::Matrix* deviceMatrix);
    void CopyHostToDevice(void* dst, const void* src, intt size);
    void CopyDeviceToHost(void* dst, const void* src, intt size);
    void CopyDeviceToDevice(void* dst, const void* src, intt size);
    void* NewDevice(intt size);
    void* NewDevice(intt size, const void* src);
    void DeleteDevice(void* devicePtr);
}

template<typename T>T* device::NewDeviceValue(T v) {
    T* valuePtr = NULL;
    void* ptr = device::NewDevice(sizeof (T));
    valuePtr = reinterpret_cast<T*> (ptr);
    device::CopyHostToDevice(valuePtr, &v, sizeof (T));
    return valuePtr;
}

template<typename T>void device::DeleteDeviceValue(T* valuePtr) {
    device::DeleteDevice(valuePtr);
}


#endif	/* MATRIXMEM_H */

