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

class CuDevice {
public:
    CuDevice();
    virtual ~CuDevice();
    virtual void setDevice(CUdevice cuDecive) = 0;
    virtual void setDeviceInfo(const CuDevice& deviceInfo) = 0;
    virtual CUdevice getDevice() const = 0;
};

class DefaultDeviceInfo : public CuDevice {
    CUdevice m_cuDevice;
public:
    DefaultDeviceInfo();
    DefaultDeviceInfo(const DefaultDeviceInfo& orig);
    virtual ~DefaultDeviceInfo();
    void setDevice(CUdevice cuDecive);
    void setDeviceInfo(const CuDevice& deviceInfo);
    CUdevice getDevice() const;
    void getDeviceProperties(CUdevprop& cuDevprop) const;
    uint getMaxThreadsPerBlock() const;
    uint getMaxThreadsX() const;
    uint getMaxThreadsY() const;
    uint getMaxBlocksX() const;
    uint getMaxBlocksY() const;
    uint getSharedMemorySize() const;
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
    void realeseImage();
    void resetParameters();
public:
    Kernel();
    virtual ~Kernel();
    uint getThreadsX() const;
    uint getThreadsY() const;
    uint getBlocksX() const;
    uint getBlocksY() const;
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

    void calculateThreadsBlocks(uintt blocks[2], uintt threads[2],
        uintt w, uintt h);

    static void SetThreadsBlocks(uintt blocks[2], uintt threads[2],
        uintt w, uintt h, uintt maxThreadsPerBlock);
        
    static CUresult Execute(const char* functionName,
        void** params, ::cuda::Kernel& kernel, void* image);
};

}
#endif	/* KERNELEXECUTOR_H */

