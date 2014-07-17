#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <map>

synchronization::RecursiveMutex DeviceMatrixAllocator::mutex;

inline void cuFree(CUdeviceptr ptr) {
    if (ptr != 0) {
        cuMemFree(ptr);
    }
}

DeviceMatrixAllocator::DeviceMatrixAllocator() :
MatrixAllocator(&DeviceMatrixModules::getInstance()) {
}

DeviceMatrixAllocator::~DeviceMatrixAllocator() {
}

DeviceMatrixUtils::DeviceMatrixUtils() :
MatrixUtils(&DeviceMatrixModules::getInstance()) {
}

DeviceMatrixUtils::~DeviceMatrixUtils() {
}

DeviceMatrixCopier::DeviceMatrixCopier() :
MatrixCopier(&DeviceMatrixModules::getInstance()) {
}

DeviceMatrixCopier::~DeviceMatrixCopier() {
}

HDMatrixCopier::HDMatrixCopier() :
MatrixCopier(&DeviceMatrixModules::getInstance()) {
}

HDMatrixCopier::~HDMatrixCopier() {
}

DHMatrixCopier::DHMatrixCopier() :
MatrixCopier(&DeviceMatrixModules::getInstance()) {
}

DHMatrixCopier::~DHMatrixCopier() {
}

intt DeviceMatrixUtils::getColumns(const math::Matrix* matrix) const {
    return this->getDeviceColumns(matrix);
}

intt DeviceMatrixUtils::getRows(const math::Matrix* matrix) const {
    return this->getDeviceRows(matrix);
}

bool DeviceMatrixUtils::isReMatrix(const math::Matrix* matrix) const {
    floatt* deviceptr = getReValues(matrix);
    return deviceptr != NULL;
}

bool DeviceMatrixUtils::isImMatrix(const math::Matrix* matrix) const {
    floatt* deviceptr = getImValues(matrix);
    return deviceptr != NULL;
}

void DeviceMatrixUtils::printInfo(const math::Matrix* matrix) const {
    uintt rows = getRows(matrix);
    uintt columns = getColumns(matrix);
    bool bre = isReMatrix(matrix);
    bool bim = isImMatrix(matrix);
    debug("matrix = %p,%u %u:%u, %d,%d \n", matrix, sizeof (matrix),
            rows, columns, bre, bim);
}

void DeviceMatrixAllocator::deleteMatrix(math::Matrix* matrix) {
    CUdeviceptr rePtr = getReValuesAddress(matrix);
    CUdeviceptr imPtr = getImValuesAddress(matrix);
    CUdeviceptr matrixPtr = reinterpret_cast<CUdeviceptr> (matrix);
    cuFree(matrixPtr);
    cuFree(imPtr);
    cuFree(rePtr);
}

void DeviceMatrixAllocator::lock() {
    DeviceMatrixAllocator::mutex.lock();
}

void DeviceMatrixAllocator::unlock() {
    DeviceMatrixAllocator::mutex.unlock();
}

math::Matrix* DeviceMatrixAllocator::newMatrix(intt columns, intt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr deviceMatrix = allocMatrix(true, true, columns, rows, value, value);
    debugFuncEnd();
    return reinterpret_cast<math::Matrix*> (deviceMatrix);
}

math::Matrix* DeviceMatrixAllocator::newReMatrix(intt columns, intt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr deviceMatrix = allocMatrix(true, false, columns, rows, value, value);
    debugFuncEnd();
    return reinterpret_cast<math::Matrix*> (deviceMatrix);
}

math::Matrix* DeviceMatrixAllocator::newImMatrix(intt columns, intt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr deviceMatrix = allocMatrix(false, true, columns, rows, value, value);
    debugFuncEnd();
    return reinterpret_cast<math::Matrix*> (deviceMatrix);
}

void setExisting(math::Matrix* matrix, void* ptr) {
    bool* is = (bool*)ptr;
    (*is) = true;
}

bool DeviceMatrixAllocator::isMatrix(math::Matrix* matrix) {
    return true;
}

math::Matrix* DeviceMatrixAllocator::newMatrixFromAsciiFile(const char* path) {
    return NULL;
}

math::Matrix* DeviceMatrixAllocator::newMatrixFromBinaryFile(const char* path) {
    return NULL;
}

void DeviceMatrixPrinter::getReMatrixStr(std::string& str, const math::Matrix* matrix) {
    math::Matrix* matrix1 = device::NewHostMatrixCopyOfDeviceMatrix(matrix);
    hmp.getReMatrixStr(str, matrix1);
    HostMatrixModules::GetInstance().getMatrixAllocator()->deleteMatrix(matrix1);
}

void DeviceMatrixPrinter::getImMatrixStr(std::string& str, const math::Matrix* matrix) {
    math::Matrix* matrix1 = device::NewHostMatrixCopyOfDeviceMatrix(matrix);
    hmp.getImMatrixStr(str, matrix1);
    HostMatrixModules::GetInstance().getMatrixAllocator()->deleteMatrix(matrix1);
}

void DeviceMatrixPrinter::getMatrixStr(std::string& str, const math::Matrix* matrix) {
    math::Matrix* matrix1 = device::NewHostMatrixCopyOfDeviceMatrix(matrix);
    hmp.getMatrixStr(str, matrix1);
    HostMatrixModules::GetInstance().getMatrixAllocator()->deleteMatrix(matrix1);
}

void DeviceMatrixPrinter::printReMatrix(FILE* stream, const math::Matrix* matrix) {
    std::string text = "";
    getReMatrixStr(text, matrix);
    fprintf(stream, "%s", text.c_str());
}

void DeviceMatrixPrinter::printReMatrix(const math::Matrix* matrix) {
    std::string text = "";
    getReMatrixStr(text, matrix);
    fprintf(stdout, "%s", text.c_str());
}

void DeviceMatrixPrinter::printReMatrix(const std::string& text, const math::Matrix* matrix) {
    std::string text1 = "";
    getReMatrixStr(text1, matrix);
    fprintf(stdout, "%s %s", text.c_str(), text1.c_str());
}

void DeviceMatrixPrinter::printImMatrix(FILE* stream, const math::Matrix* matrix) {
    std::string text = "";
    getImMatrixStr(text, matrix);
    fprintf(stream, "%s", text.c_str());
}

void DeviceMatrixPrinter::printImMatrix(const math::Matrix* matrix) {
    std::string text = "";
    getImMatrixStr(text, matrix);
    fprintf(stdout, "%s", text.c_str());
}

void DeviceMatrixPrinter::printImMatrix(const std::string& text, const math::Matrix* matrix) {
    std::string text1 = "";
    getImMatrixStr(text1, matrix);
    fprintf(stdout, "%s %s", text.c_str(), text1.c_str());
}

math::Matrix* DeviceMatrixModules::newDeviceMatrix(math::Matrix* hostMatrix) {
    bool allocRe = hostMatrix->reValues != NULL;
    bool allocIm = hostMatrix->imValues != NULL;
    DeviceUtils dma;
    CUdeviceptr ptr = dma.allocMatrix(allocRe, allocIm, hostMatrix->columns, hostMatrix->rows);
    math::Matrix* mptr = reinterpret_cast<math::Matrix*> (ptr);
    if (allocRe && allocIm) {
        hdmc.copyMatrixToMatrix(mptr, hostMatrix);
    } else if (allocRe) {
        hdmc.copyReMatrixToReMatrix(mptr, hostMatrix);
    } else if (allocIm) {
        hdmc.copyImMatrixToImMatrix(mptr, hostMatrix);
    }
    return mptr;
}

MatrixAllocator* DeviceMatrixModules::getMatrixAllocator() {
    return &dma;
}

MatrixCopier* DeviceMatrixModules::getMatrixCopier() {
    return &dmc;
}

MatrixUtils* DeviceMatrixModules::getMatrixUtils() {
    return &dmu;
}

MatrixPrinter* DeviceMatrixModules::getMatrixPrinter() {
    return &dmp;
}

HDMatrixCopier* DeviceMatrixModules::getHDCopier() {
    return &hdmc;
}

DHMatrixCopier* DeviceMatrixModules::getDHCopier() {
    return &dhmc;
}

DeviceMatrixModules::DeviceMatrixModules() {
}

DeviceMatrixModules::~DeviceMatrixModules() {
}

DeviceMatrixModules DeviceMatrixModules::deviceMatrixMoules;

DeviceMatrixModules& DeviceMatrixModules::getInstance() {
    return deviceMatrixMoules;
}

DeviceMatrixPrinter::DeviceMatrixPrinter() :
MatrixPrinter(&DeviceMatrixModules::getInstance()) {
}

DeviceMatrixPrinter::~DeviceMatrixPrinter() {
}

void DeviceMatrixUtils::setDiagonalReMatrix(math::Matrix* matrix, floatt value) {
}

void DeviceMatrixUtils::setDiagonalImMatrix(math::Matrix* matrix, floatt value) {
}

void DeviceMatrixUtils::setZeroReMatrix(math::Matrix* matrix) {
}

void DeviceMatrixUtils::setZeroImMatrix(math::Matrix* matrix) {
}

namespace device {

    math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix) {
        DeviceUtils deviceUtils;
        CUdeviceptr matrixRePtr = deviceUtils.getReValuesAddress(matrix);
        CUdeviceptr matrixImPtr = deviceUtils.getImValuesAddress(matrix);
        uintt columns = deviceUtils.getDeviceColumns(matrix);
        uintt rows = deviceUtils.getDeviceRows(matrix);
        math::Matrix * matrix1 = NULL;
        if (matrixRePtr != 0 && matrixImPtr != 0) {
            matrix1 = HostMatrixModules::GetInstance().getMatrixAllocator()->newMatrix(columns, rows);
        } else if (matrixRePtr != 0) {
            matrix1 = HostMatrixModules::GetInstance().getMatrixAllocator()->newReMatrix(columns, rows);
        } else if (matrixImPtr != 0) {
            matrix1 = HostMatrixModules::GetInstance().getMatrixAllocator()->newImMatrix(columns, rows);
        }
        DeviceMatrixModules::getInstance().getDHCopier()->copyMatrixToMatrix(matrix1, matrix);
        return matrix1;
    }

    math::Matrix* NewDeviceMatrix(math::Matrix* hostMatrix) {
        return DeviceMatrixModules::getInstance().newDeviceMatrix(hostMatrix);
    }

    void DeleteDeviceMatrix(math::Matrix* deviceMatrix) {
        DeviceMatrixModules::getInstance().getMatrixAllocator()->deleteMatrix(deviceMatrix);
    }

    void CopyDeviceMatrixToHostMatrix(math::Matrix* hostMatrix, math::Matrix* deviceMatrix) {
        DeviceMatrixModules::getInstance().getDHCopier()->copyMatrixToMatrix(hostMatrix, deviceMatrix);
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

    void* NewDevice(intt size) {
        CUdeviceptr devicePtr;
        cuMemAlloc(&devicePtr, size);
        cuMemsetD32(devicePtr, 0, size);
        return reinterpret_cast<void*> (devicePtr);
    }

    void* NewDevice(intt size, const void* src) {
        void* devPtr = NewDevice(size);
        CopyHostToDevice(devPtr, src, size);
        return devPtr;
    }

    void DeleteDevice(void* devicePtr) {
        if (devicePtr) {
            CUdeviceptr deviecPtr = reinterpret_cast<CUdeviceptr> (devicePtr);
            cuFree(deviecPtr);
        }
    }
}