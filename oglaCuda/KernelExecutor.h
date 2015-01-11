/* 
 * File:   KernelExecutor.h
 * Author: mmatula
 *
 * Created on January 19, 2014, 9:47 AM
 */

#ifndef OGLA_KERNEL_EXECUTOR_H
#define	OGLA_KERNEL_EXECUTOR_H

#include <cuda.h>
#include <string>
#include "Math.h"
//#include "MatrixStructure.h"

#define printCuError(cuResult) if(cuResult != 0) {debug("\n\n %s %s : %d cuError == %d \n\n",__FUNCTION__,__FILE__,__LINE__,cuResult); }
#define printCuErrorStatus(status, cuResult) if(cuResult != 0) {status = cuResult; debug("\n\n %s %s : %d cuError == %d \n\n",__FUNCTION__,__FILE__,__LINE__,cuResult); }

namespace cuda {

void Init();

class DeviceInfo {
public:
    DeviceInfo();
    virtual ~DeviceInfo();
    virtual void setDevice(CUdevice cuDecive) = 0;
    virtual void setDeviceInfo(const DeviceInfo& deviceInfo) = 0;
    virtual CUdevice getDevice() const = 0;
};

class DefaultDeviceInfo : public DeviceInfo {
    CUdevice m_cuDevice;
public:
    DefaultDeviceInfo();
    DefaultDeviceInfo(const DefaultDeviceInfo& orig);
    virtual ~DefaultDeviceInfo();
    void setDevice(CUdevice cuDecive);
    void setDeviceInfo(const DeviceInfo& deviceInfo);
    CUdevice getDevice() const;
    void getDeviceProperties(CUdevprop& cuDevprop) const;
    uint getMaxThreadsPerBlock() const;
    uint getThreadsX() const;
    uint getThreadsY() const;
    uint getBlocksX() const;
    uint getBlocksY() const;
};

class Context : public DefaultDeviceInfo {
public:
    void init();
    void destroy();
    static Context& Instance();
protected:
    Context(int deviceIndex = 1);
    virtual ~Context();
private:
    static Context m_Context;
    CUcontext context;
    int deviceIndex;
};

class Kernel : public DefaultDeviceInfo {
    void** m_params;
    int m_paramsSize;
    void* m_image;
    std::string m_path;
    uintt m_threadsCount[3];
    uintt m_blocksCount[3];
    uintt m_sharedMemoryInBytes;
    inline void realeseImage();
public:
    Kernel();
    virtual ~Kernel();
    static void* LoadImage(const char* path);
    static void* LoadImage(const char** pathes);
    static void FreeImage(void* image);
    Kernel(void* image, CUdevice cuDevicePtr);
    void setThreadsCount(intt x, intt y);
    void setBlocksCount(intt x, intt y);
    void setDimensions(uintt w, uintt h);
    void setSharedMemory(uintt sizeInBytes);
    void setParams(void** params);
    int getParamsCount() const;
    void** getParams() const;
    void setImage(void* image);
    bool loadImage(const char* path);
    bool loadImage(const char** pathes);
    virtual CUresult execute(const char* functionName);

    static CUresult ExecuteKernel(const char* functionName,
        void** params, ::cuda::Kernel& kernel, void* image);
};

}
#endif	/* KERNELEXECUTOR_H */

