/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef OAP_KERNEL_EXECUTOR_H
#define	OAP_KERNEL_EXECUTOR_H

#include "CuCore.h"
#include <string>
#include <stack>
#include "Math.h"
#include "Matrix.h"
#include "CudaUtils.h"
//#include "MatrixStructure.h"

#define printCuError(cuResult) if(cuResult != 0) {debug("\n\n %s %s : %d cuError == %d \n\n",__FUNCTION__,__FILE__,__LINE__,cuResult); abort();}
#define printCuErrorStatus(status, cuResult) if(cuResult != 0) {status = cuResult; debug("\n\n %s %s : %d cuError == %d \n\n",__FUNCTION__,__FILE__,__LINE__,cuResult); abort();}

namespace device {

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
    void create();
    void destroy();
    static Context& Instance();
protected:
    Context(int deviceIndex = 1);
    virtual ~Context();
private:
    static Context m_Context;
    std::stack<CUcontext> m_contexts;
    int deviceIndex;
};

class Kernel : public DefaultDeviceInfo {
    void** m_params;
    int m_paramsSize;
    void* m_image;
    std::string m_path;
    CUmodule m_cuModule;
    uintt m_threadsCount[3];
    uintt m_blocksCount[3];
    uintt m_sharedMemoryInBytes;
    void releaseImage();
    void resetParameters();
    void unloadCuModule();
    void loadCuModule();
    void setImage(void* image);
public:
    Kernel();
    virtual ~Kernel();
    uint getThreadsX() const;
    uint getThreadsY() const;
    uint getBlocksX() const;
    uint getBlocksY() const;
    static void* LoadImage(const char* path);
    static void* LoadImage(std::string& path, const char** pathes);
    static void FreeImage(void* image);
    void setThreadsCount(intt x, intt y);
    void setBlocksCount(intt x, intt y);
    void setDimensions(uintt w, uintt h);
    void setDimensionsDevice(math::Matrix* dmatrix);
    void setSharedMemory(uintt sizeInBytes);
    void setParams(void** params);
    int getParamsCount() const;
    void** getParams() const;
    bool load(const char* path);
    bool load(const char** pathes);
    void unload();
    CUresult execute(const char* functionName);

    void calculateThreadsBlocks(uintt blocks[2], uintt threads[2],
        uintt w, uintt h);

    void calculateThreadsBlocksDevice(uintt blocks[2], uintt threads[2],
        math::Matrix* dmatrix);

    static void SetThreadsBlocks(uintt blocks[2], uintt threads[2],
        uintt w, uintt h, uintt maxThreadsPerBlock);
        
    static CUresult Execute(const char* functionName,
        void** params, ::device::Kernel& kernel);
};

}
#endif	/* KERNELEXECUTOR_H */
