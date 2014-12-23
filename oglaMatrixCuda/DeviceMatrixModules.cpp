#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <map>

synchronization::RecursiveMutex DeviceMatrixAllocator::mutex;

DeviceMatrixAllocator::DeviceMatrixAllocator() :
MatrixAllocator(DeviceMatrixModules::GetInstance()) {
}

DeviceMatrixAllocator::~DeviceMatrixAllocator() {
}

DeviceMatrixUtils::DeviceMatrixUtils() :
MatrixUtils(DeviceMatrixModules::GetInstance()) {
}

DeviceMatrixUtils::~DeviceMatrixUtils() {
}

DeviceMatrixCopier::DeviceMatrixCopier() :
MatrixCopier(DeviceMatrixModules::GetInstance()) {
}

DeviceMatrixCopier::~DeviceMatrixCopier() {
}

HDMatrixCopier::HDMatrixCopier() :
MatrixCopier(DeviceMatrixModules::GetInstance()) {
}

HDMatrixCopier::~HDMatrixCopier() {
}

DHMatrixCopier::DHMatrixCopier() :
MatrixCopier(DeviceMatrixModules::GetInstance()) {
}

DHMatrixCopier::~DHMatrixCopier() {
}

uintt DeviceMatrixUtils::getColumns(const math::Matrix* matrix) const {
    return CudaUtils::GetDeviceColumns(matrix);
}

uintt DeviceMatrixUtils::getRows(const math::Matrix* matrix) const {
    return CudaUtils::GetDeviceRows(matrix);
}

bool DeviceMatrixUtils::isReMatrix(const math::Matrix* matrix) const {
    floatt* deviceptr = CudaUtils::GetReValues(matrix);
    return deviceptr != NULL;
}

bool DeviceMatrixUtils::isImMatrix(const math::Matrix* matrix) const {
    floatt* deviceptr = CudaUtils::GetImValues(matrix);
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
    CUdeviceptr rePtr = CudaUtils::GetReValuesAddress(matrix);
    CUdeviceptr imPtr = CudaUtils::GetImValuesAddress(matrix);
    CUdeviceptr matrixPtr = reinterpret_cast<CUdeviceptr> (matrix);
    CudaUtils::DeleteDevice(matrixPtr);
    CudaUtils::DeleteDevice(imPtr);
    CudaUtils::DeleteDevice(rePtr);
}

void DeviceMatrixAllocator::lock() {
    DeviceMatrixAllocator::mutex.lock();
}

void DeviceMatrixAllocator::unlock() {
    DeviceMatrixAllocator::mutex.unlock();
}

math::Matrix* DeviceMatrixAllocator::newMatrix(uintt columns, uintt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr deviceMatrix = CudaUtils::AllocMatrix(true, true, columns, rows, value, value);
    debugFuncEnd();
    return reinterpret_cast<math::Matrix*> (deviceMatrix);
}

math::Matrix* DeviceMatrixAllocator::newReMatrix(uintt columns, uintt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr deviceMatrix = CudaUtils::AllocMatrix(true, false, columns, rows, value, value);
    debugFuncEnd();
    return reinterpret_cast<math::Matrix*> (deviceMatrix);
}

math::Matrix* DeviceMatrixAllocator::newImMatrix(uintt columns, uintt rows, floatt value) {
    debugFuncBegin();
    CUdeviceptr deviceMatrix = CudaUtils::AllocMatrix(false, true, columns, rows, value, value);
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
    math::Matrix* matrix1 = cuda::NewHostMatrixCopyOfDeviceMatrix(matrix);
    hmp.getReMatrixStr(str, matrix1);
    HostMatrixModules::GetInstance()->getMatrixAllocator()->deleteMatrix(matrix1);
}

void DeviceMatrixPrinter::getImMatrixStr(std::string& str, const math::Matrix* matrix) {
    math::Matrix* matrix1 = cuda::NewHostMatrixCopyOfDeviceMatrix(matrix);
    hmp.getImMatrixStr(str, matrix1);
    HostMatrixModules::GetInstance()->getMatrixAllocator()->deleteMatrix(matrix1);
}

void DeviceMatrixPrinter::getMatrixStr(std::string& str, const math::Matrix* matrix) {
    math::Matrix* matrix1 = cuda::NewHostMatrixCopyOfDeviceMatrix(matrix);
    hmp.getMatrixStr(str, matrix1);
    HostMatrixModules::GetInstance()->getMatrixAllocator()->deleteMatrix(matrix1);
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
    return cuda::NewDeviceMatrix(hostMatrix);
}

MatrixAllocator* DeviceMatrixModules::getMatrixAllocator() {
    return m_dma;
}

MatrixCopier* DeviceMatrixModules::getMatrixCopier() {
    return m_dmc;
}

MatrixUtils* DeviceMatrixModules::getMatrixUtils() {
    return m_dmu;
}

MatrixPrinter* DeviceMatrixModules::getMatrixPrinter() {
    return m_dmp;
}

HDMatrixCopier* DeviceMatrixModules::getHDCopier() {
    return m_hdmc;
}

DHMatrixCopier* DeviceMatrixModules::getDHCopier() {
    return m_dhmc;
}

DeviceMatrixModules::DeviceMatrixModules() {
    m_dhmc = NULL;
    m_dma = NULL;
    m_dmc = NULL;
    m_dmp = NULL;
    m_dmu = NULL;
    m_hdmc = NULL;
}

DeviceMatrixModules::~DeviceMatrixModules() {
}

DeviceMatrixModules* DeviceMatrixModules::deviceMatrixMoules = NULL;

DeviceMatrixModules* DeviceMatrixModules::GetInstance() {
    if (NULL == deviceMatrixMoules) {
        deviceMatrixMoules = new DeviceMatrixModules();
        deviceMatrixMoules->m_dma = new DeviceMatrixAllocator;
        deviceMatrixMoules->m_dmc = new DeviceMatrixCopier;
        deviceMatrixMoules->m_dmu = new DeviceMatrixUtils;
        deviceMatrixMoules->m_dmp = new DeviceMatrixPrinter;
        deviceMatrixMoules->m_hdmc = new HDMatrixCopier;
        deviceMatrixMoules->m_dhmc = new DHMatrixCopier;
    }
    return deviceMatrixMoules;
}

DeviceMatrixPrinter::DeviceMatrixPrinter() :
MatrixPrinter(DeviceMatrixModules::GetInstance()) {
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

namespace cuda {

    math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix) {
        CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
        CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
        uintt columns = CudaUtils::GetDeviceColumns(matrix);
        uintt rows = CudaUtils::GetDeviceRows(matrix);
        math::Matrix * matrix1 = NULL;
        if (matrixRePtr != 0 && matrixImPtr != 0) {
            matrix1 = HostMatrixModules::GetInstance()->getMatrixAllocator()->newMatrix(columns, rows);
        } else if (matrixRePtr != 0) {
            matrix1 = HostMatrixModules::GetInstance()->getMatrixAllocator()->newReMatrix(columns, rows);
        } else if (matrixImPtr != 0) {
            matrix1 = HostMatrixModules::GetInstance()->getMatrixAllocator()->newImMatrix(columns, rows);
        }
        DeviceMatrixModules::GetInstance()->getDHCopier()->copyMatrixToMatrix(matrix1, matrix);
        return matrix1;
    }

    math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix) {
        return NewDeviceMatrix(hostMatrix, hostMatrix->columns, hostMatrix->rows);
    }

    math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix,
            uintt columns, uintt rows) {
        bool allocRe = hostMatrix->reValues != NULL;
        bool allocIm = hostMatrix->imValues != NULL;
        CUdeviceptr ptr = CudaUtils::AllocMatrix(allocRe, allocIm,
                columns, rows);
        math::Matrix* mptr = reinterpret_cast<math::Matrix*> (ptr);
        return mptr;
    }

    math::Matrix* NewDeviceMatrix(uintt columns, uintt rows) {
        return DeviceMatrixModules::GetInstance()->getMatrixAllocator()->newMatrix(columns, rows);
    }

    void DeleteDeviceMatrix(math::Matrix* deviceMatrix) {
        DeviceMatrixModules::GetInstance()->getMatrixAllocator()->deleteMatrix(deviceMatrix);
    }

    void CopyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src) {
        uintt length1 = dst->columns * dst->rows;
        uintt length2 = CudaUtils::GetDeviceColumns(src) * CudaUtils::GetDeviceRows(src);
        length1 = length1 < length2 ? length1 : length2;
        CUdeviceptr srcRePtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetReValues(src));
        CUdeviceptr srcImPtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetImValues(src));
        if (srcRePtr != 0 && dst->reValues != NULL) {
            cuMemcpyDtoH(dst->reValues, srcRePtr, length1 * sizeof (floatt));
        }
        if (srcImPtr != 0 && dst->imValues != NULL) {
            cuMemcpyDtoH(dst->imValues, srcImPtr, length1 * sizeof (floatt));
        }
    }

    void CopyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src) {
        uintt length1 = CudaUtils::GetDeviceColumns(dst) * CudaUtils::GetDeviceRows(dst);
        uintt length2 = src->columns * src->rows;
        length1 = length1 < length2 ? length1 : length2;
        CUdeviceptr dstRePtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetReValues(dst));
        CUdeviceptr dstImPtr = reinterpret_cast<CUdeviceptr> (CudaUtils::GetImValues(dst));
        if (dstRePtr != 0 && src->reValues != NULL) {
            cuMemcpyHtoD(dstRePtr, src->reValues, length1 * sizeof (floatt));
        }
        if (dstImPtr != 0 && src->imValues != NULL) {
            cuMemcpyHtoD(dstImPtr, src->imValues, length1 * sizeof (floatt));
        }
    }

    void CopyHostArraysToDeviceMatrix(math::Matrix* dst, const floatt* rearray,
            const floatt* imarray) {
        uintt columns = CudaUtils::GetDeviceColumns(dst);
        uintt rows = CudaUtils::GetDeviceRows(dst);
        uintt length1 = columns * rows;
        math::Matrix matrix = {
            columns, rows,
            const_cast<floatt*> (rearray),
            const_cast<floatt*> (imarray),
            columns, rows
        };
        CopyHostMatrixToDeviceMatrix(dst, &matrix);
    }
}