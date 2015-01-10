/* 
 * File:   KernelExecutor.cpp
 * Author: mmatula
 * 
 * Created on January 19, 2014, 9:47 AM
 */

#include <linux/fs.h>
#include <math.h>

#include "KernelExecutor.h"
#include "ArrayTools.h"
#include "DebugLogs.h"
#include "ThreadsMapper.h"

#define KERNEL_EXECUTOR_LOGS_FUNCTION_NAME
#define KERNEL_EXECUTOR_LOGS

namespace cuda {

void PrintDeviceInfo(CUdevice cudevice) {
    CUdevprop cuDevprop;
    printCuError(cuDeviceGetProperties(&cuDevprop, cudevice));
    debug("Device properties: \n --Max grid size: %d, %d, %d.\n --Max threads dim: %d, %d, %d",
        cuDevprop.maxGridSize[0], cuDevprop.maxGridSize[1], cuDevprop.maxGridSize[2],
        cuDevprop.maxThreadsDim[0], cuDevprop.maxThreadsDim[1], cuDevprop.maxThreadsDim[2]);
    debug(" --Max threads per block: %d", cuDevprop.maxThreadsPerBlock);
    debug(" --Register per block: %d", cuDevprop.regsPerBlock);
    debug(" --Shared memory per block: %d", cuDevprop.sharedMemPerBlock);
}

void Init() {
    static bool wasInit = false;
    debugFuncBegin();
    if (wasInit == false) {
        wasInit = true;
        debugFunc();
        printCuError(cuInit(0));
    }
    debugFuncEnd();
}

DeviceInfo::DeviceInfo() {
}

DeviceInfo::~DeviceInfo() {
}

DefaultDeviceInfo::DefaultDeviceInfo() : m_cuDevice(0) {
}

DefaultDeviceInfo::DefaultDeviceInfo(const DefaultDeviceInfo& orig) :
m_cuDevice(orig.m_cuDevice) {
}

DefaultDeviceInfo::~DefaultDeviceInfo() {
    m_cuDevice = 0;
}

CUdevice DefaultDeviceInfo::getDevice() const {
    return m_cuDevice;
}

void DefaultDeviceInfo::getDeviceProperties(CUdevprop& cuDevprop) {
    printCuError(cuDeviceGetProperties(&cuDevprop, m_cuDevice));
}

void DefaultDeviceInfo::setDevice(CUdevice cuDecive) {
    debugFuncBegin();
    debug("%p %d \n", this, cuDecive);
    //PrintDeviceInfo(cuDecive);
    this->m_cuDevice = m_cuDevice;
    debugFuncEnd();
}

void DefaultDeviceInfo::setDeviceInfo(const DeviceInfo& deviceInfo) {
    setDevice(deviceInfo.getDevice());
}

Context::Context(int _deviceIndex) :
context(NULL), deviceIndex(_deviceIndex) {
}

Context Context::m_Context;

Context& Context::Instance() {
    return Context::m_Context;
}

void Context::init() {
    Init();
    if (context == NULL) {
        int count = 2;
        printCuError(cuDeviceGetCount(&count));
        debug("Devices count: %d \n", count);
        if (deviceIndex < count) {
            CUdevice device = 0;
            printCuError(cuDeviceGet(&device, deviceIndex));
            setDevice(device);
            printCuError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
        }
    }
}

void Context::destroy() {
    if (context != NULL) {
        printCuError(cuCtxDestroy(context));
        context = NULL;
    }
}

Context::~Context() {
    destroy();
}

void Kernel::setThreadsCount(intt x, intt y) {
    m_threadsCount[0] = x;
    m_threadsCount[1] = y;
    m_threadsCount[2] = 1;
}

void Kernel::setBlocksCount(intt x, intt y) {
    m_blocksCount[0] = x;
    m_blocksCount[1] = y;
    m_blocksCount[2] = 1;
}

void Kernel::setDimensions(uintt w, uintt h) {
    CUdevprop devprop;
    getDeviceProperties(devprop);
    utils::mapper::SetThreadsBlocks(m_blocksCount, m_threadsCount, w, h,
        devprop.maxThreadsPerBlock);
}

void Kernel::setSharedMemory(uintt sizeInBytes) {
    m_sharedMemoryInBytes = sizeInBytes;
}

void Kernel::setParams(void** params) {
    m_params = params;
}

Kernel::Kernel() : m_params(NULL), m_paramsSize(0) {
    this->m_blocksCount[0] = 1;
    this->m_blocksCount[1] = 1;
    this->m_blocksCount[2] = 1;
    this->m_threadsCount[0] = 1;
    this->m_threadsCount[1] = 1;
    this->m_threadsCount[2] = 1;
    m_sharedMemoryInBytes = 0;
    this->m_image = NULL;
}

Kernel::Kernel(void* image, CUdevice cuDevicePtr) :
m_params(NULL),
m_paramsSize(0) {
    this->m_blocksCount[0] = 1;
    this->m_blocksCount[1] = 1;
    this->m_threadsCount[0] = 1;
    this->m_threadsCount[1] = 1;
    m_sharedMemoryInBytes = 0;
    this->m_image = image;
}

int Kernel::getParamsCount() const {
    return this->m_paramsSize;
}

void** Kernel::getParams() const {
    return this->m_params;
}

void Kernel::setImage(void* image) {
    this->m_image = image;
}

void Kernel::execute(const char* functionName) {
    if (NULL == m_image && m_path.length() == 0) {
        debugError("Error: image and path not defined. Function name: %s \n", functionName);
    }
    CUmodule cuModule = NULL;
    CUfunction cuFunction = NULL;
#ifdef KERNEL_EXECUTOR_LOGS_FUNCTION_NAME
    debug("Function name: %s", functionName);
#endif
    if (NULL != m_image) {
        printCuError(cuModuleLoadData(&cuModule, m_image));
    } else if (m_path.length() > 0) {
        printCuError(cuModuleLoad(&cuModule, m_path.c_str()));
    }
    if (NULL != cuModule) {
        printCuError(cuModuleGetFunction(&cuFunction, cuModule, functionName));
#ifdef KERNEL_EXECUTOR_LOGS
        debug("Load kernel: %s", functionName);
        debug("Image: %p", m_image);
        debug("Module handle: %p", cuModule);
        debug("Function handle: %p", cuFunction);
        PrintDeviceInfo(getDevice());
        debug(" Execution:");
        debug(" --threads dim: %d, %d, %d", m_threadsCount[0], m_threadsCount[1], m_threadsCount[2]);
        debug(" --grid size: %d, %d, %d", m_blocksCount[0], m_blocksCount[1], m_blocksCount[2]);
#endif
        printCuError(cuLaunchKernel(cuFunction,
            m_blocksCount[0], m_blocksCount[1], m_blocksCount[2],
            m_threadsCount[0], m_threadsCount[1], m_threadsCount[2],
            m_sharedMemoryInBytes, NULL, this->getParams(), NULL));
    } else {
        if (NULL != m_image) {
            debug("Module is incorrect %p %p;", cuModule, m_image);
        } else if (m_path.length() > 0) {
            debug("Module is incorrect %d %s;", cuModule, m_path.c_str());
        }
    }
}

Kernel::~Kernel() {
}

inline char* loadData(FILE* f) {
    if (f) {
        fseek(f, 0, SEEK_END);
        long int size = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* data = new char[size];
        memset(data, 0, size);
        fread(data, size, 1, f);
        fclose(f);
        return data;
    }
    return NULL;
}

inline char* loadData(const char** pathes) {
    while (pathes != NULL && *pathes != NULL) {
        FILE* f = fopen(*pathes, "rb");
        if (f != NULL) {
            char* data = loadData(f);
            return data;
        }
        ++pathes;
    }
    return NULL;
}

inline char* loadData(const char* path) {
    const char* pathes[] = {path, NULL};
    return loadData(pathes);
}

void* Kernel::LoadImage(const char* path) {
    return loadData(path);
}

void* Kernel::LoadImage(const char** pathes) {
    return loadData(pathes);
}

void Kernel::FreeImage(void* image) {
    char* data = reinterpret_cast<char*> (image);
    delete[] data;
}

bool Kernel::loadImage(const char* path) {
    m_image = loadData(path);
#ifdef DEBUG
    if (m_image == NULL) {
        debug("Image with path: %s, doens't exist.\n", path);
    } else {
        debug("Image with path: %s, exist.\n", path);
    }
#endif
    return m_image != NULL;
}

bool Kernel::loadImage(const char** pathes) {
    m_image = loadData(pathes);
#ifdef DEBUG
    if (m_image == NULL) {
        debug("Image with path:, doens't exist.\n");
    } else {
        debug("Image with path: , exist\n");
    }
#endif
    return m_image != NULL;
}

void Kernel::realeseImage() {
    if (m_image) {
        char* data = (char*) m_image;
        delete[] data;
    }
}

void KernelMatrix::setMatrixSizes(uintt width, uintt height) {
    m_width = width;
    m_height = height;
}

KernelMatrix::KernelMatrix() : m_width(0), m_height(0) {
}

KernelMatrix::~KernelMatrix() {
}

void KernelMatrix::getThreadsBlocks(int threads[2], int blocks[2]) {
    if (m_width > 0 && m_height > 0) {
        CUdevprop prop;
        getDeviceProperties(prop);
        int threadsX = m_width;
        int threadsY = m_height;
        int blocksX = 1;
        int blocksY = 1;
        int m = threadsX * threadsY;
        if (m > prop.maxThreadsPerBlock) {
            floatt factor = 2;
            if (m % 2 == 0) {
                factor = 2;
            } else if (m % 3 == 0) {
                factor = 3;
            } else if (m % 5 == 0) {
                factor = 5;
            }
            while (threadsX * threadsY > prop.maxThreadsPerBlock) {
                threadsX = threadsX / factor;
                threadsY = threadsY / factor;
                blocksX = blocksX * factor;
                blocksY = blocksY * factor;
            }
        }
        threads[0] = threadsX;
        threads[1] = threadsY;
        blocks[0] = blocksX;
        blocks[1] = blocksY;
    } else {
        threads[0] = 0;
        threads[1] = 0;
        blocks[0] = 0;
        blocks[1] = 0;
    }
}

void KernelMatrix::execute(const char* functionName) {
    debugFuncBegin();
    int threads[2];
    int blocks[2];
    getThreadsBlocks(threads, blocks);
    if (threads[0] != 0) {
        setThreadsCount(threads[0], threads[1]);
        setBlocksCount(blocks[0], blocks[1]);
        Kernel::execute(functionName);
        debugFuncEnd();
    } else {
        abort();
    }
}

void Kernel::ExecuteKernel(const char* functionName,
    void** params, ::cuda::Kernel& kernel, void* image) {
    kernel.setImage(image);
    kernel.setParams(params);
    kernel.execute(functionName);
}
}