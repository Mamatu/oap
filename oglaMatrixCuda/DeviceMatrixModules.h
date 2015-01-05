#ifndef OGLA_DEVICE_MATRIX_UTILS_H
#define	OGLA_DEVICE_MATRIX_UTILS_H
#include <stdio.h>
#include <cuda.h>
#include <map>
#include "HostMatrixModules.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "ThreadUtils.h"
#include "CudaUtils.h"

class DeviceMatrixAllocator : public MatrixAllocator {
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
    math::Matrix* newReMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* newImMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* newMatrix(uintt columns, uintt rows, floatt value = 0);
    bool isMatrix(math::Matrix* matrix);
    math::Matrix* newMatrixFromAsciiFile(const char* path);
    math::Matrix* newMatrixFromBinaryFile(const char* path);
    void deleteMatrix(math::Matrix* matrix);
};

class DeviceMatrixUtils : public MatrixUtils {
    DeviceMatrixAllocator dma;
public:
    DeviceMatrixUtils();
    ~DeviceMatrixUtils();
    void setDiagonalReMatrix(math::Matrix* matrix, floatt value);
    void setDiagonalImMatrix(math::Matrix* matrix, floatt value);
    void setZeroReMatrix(math::Matrix* matrix);
    void setZeroImMatrix(math::Matrix* matrix);
    uintt getColumns(const math::Matrix* matrix) const;
    uintt getRows(const math::Matrix* matrix) const;
    bool isReMatrix(const math::Matrix* matrix) const;
    bool isImMatrix(const math::Matrix* matrix) const;
    void printInfo(const math::Matrix* matrix) const;
};

class DeviceMatrixCopier : public MatrixCopier {
public:
    DeviceMatrixCopier();
    virtual ~DeviceMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    void copy(floatt* dst, const floatt* src, uintt length);
    void setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column);
};

/**
 * This class allows to copy from Host to Device.
 */
class HDMatrixCopier : public MatrixCopier {
public:
    HDMatrixCopier();
    virtual ~HDMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    void copy(floatt* dst, const floatt* src, uintt length);
    void setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column);
};

/**
 * This class allows to copy from Device to Host.
 */
class DHMatrixCopier : public MatrixCopier {
public:
    DHMatrixCopier();
    virtual ~DHMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    void copy(floatt* dst, const floatt* src, uintt length);
    void setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);

    void setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column);
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
    DeviceMatrixAllocator* m_dma;
    DeviceMatrixCopier* m_dmc;
    DeviceMatrixUtils* m_dmu;
    DeviceMatrixPrinter* m_dmp;
    HDMatrixCopier* m_hdmc;
    DHMatrixCopier* m_dhmc;
    static DeviceMatrixModules* deviceMatrixMoules;
protected:
    DeviceMatrixModules();
    virtual ~DeviceMatrixModules();
public:
    math::Matrix* newDeviceMatrix(math::Matrix* hostMatrix);
    static DeviceMatrixModules* GetInstance();
    virtual MatrixAllocator* getMatrixAllocator();
    virtual MatrixCopier* getMatrixCopier();
    virtual HDMatrixCopier* getHDCopier();
    virtual DHMatrixCopier* getDHCopier();
    virtual MatrixUtils* getMatrixUtils();
    virtual MatrixPrinter* getMatrixPrinter();
};

namespace cuda {

math::Matrix* NewDeviceMatrix(uintt columns, uintt rows);

math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix);

math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix,
    uintt columns, uintt rows);

math::Matrix* NewDeviceMatrix(bool allocRe, bool allocIm,
    uintt columns, uintt rows);

math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix);

void DeleteDeviceMatrix(math::Matrix* deviceMatrix);

/**
 * 
 * @param dst
 * @param src
 */
void CopyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * 
 * @param dst
 * @param src
 */
void CopyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src);

void CopyDeviceMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * 
 * @param dst
 * @param src
 */
void CopyHostArraysToDeviceMatrix(math::Matrix* dst, const floatt* rearray,
    const floatt* imarray);

MatrixEx** NewDeviceMatrixEx(uintt count);
void DeleteDeviceMatrixEx(MatrixEx** matrixEx);
void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count);

MatrixEx* NewDeviceMatrixEx();
void DeleteDeviceMatrixEx(MatrixEx* matrixEx);
void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx);

void PrintReMatrix(FILE* stream, const math::Matrix* matrix);
void PrintReMatrix(const math::Matrix* matrix);
void PrintReMatrix(const std::string& text, const math::Matrix* matrix);
void PrintImMatrix(FILE* stream, const math::Matrix* matrix);
void PrintImMatrix(const math::Matrix* matrix);
void PrintImMatrix(const std::string& text, const math::Matrix* matrix);

}


#endif	/* MATRIXMEM_H */

