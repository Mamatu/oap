/* 
 * File:   KernelExecutor.h
 * Author: mmatula
 *
 * Created on January 19, 2014, 9:47 AM
 */

#ifndef OGLA_KERNEL_EXECUTOR_H
#define	OGLA_KERNEL_EXECUTOR_H

#include <cuda.h>
#include "Math.h"
#include "DeviceMatrixModules.h"
//#include "MatrixStructure.h"

#define printCuError(cuResult) debug("%s %s : %d cuError == %d \n",__FUNCTION__,__FILE__,__LINE__,cuResult);

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
        void getDeviceProperties(CUdevprop& cuDevprop);
    };

    class Context : public DefaultDeviceInfo {
        CUcontext context;
        int deviceIndex;
    public:
        Context(int deviceIndex = 0);
        virtual ~Context();
        void init();
        void destroy();
    };

    class Kernel : public DefaultDeviceInfo {
        void** m_params;
        int m_paramsSize;
        void* m_image;
        std::string m_path;
        intt m_threadsCount[3];
        intt m_blocksCount[3];
        inline void realeseImage();
    protected:
        void getDeviceProperties(CUdevprop& cuDevprop, CUdevice* cuDevicePtr);
    public:
        Kernel();
        virtual ~Kernel();
        static void* LoadImage(const char* path);
        static void* LoadImage(const char** pathes);
        static void FreeImage(void* image);
        Kernel(void* image, CUdevice cuDevicePtr);
        void setThreadsCount(intt x, intt y);
        void setBlocksCount(intt x, intt y);
        int sharedMemorySize;
        void setParams(void** params);
        int getParamsCount() const;
        void** getParams() const;
        void setImage(void* image);
        bool loadImage(const char* path);
        bool loadImage(const char** pathes);
        virtual void execute(const char* functionName);
    };

    class KernelMatrix : public Kernel {
        uintt m_width;
        uintt m_height;
    public:
        void getThreadsBlocks(int threads[2], int blokcs[2]);
        void setMatrixSizes(uintt width, uintt height);
        KernelMatrix();
        virtual ~KernelMatrix();
        void execute(const char* functionName);
    };
}
#endif	/* KERNELEXECUTOR_H */

