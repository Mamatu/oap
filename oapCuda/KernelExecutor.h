/*
 * Copyright 2016 - 2021 Marcin Matula
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
#define OAP_KERNEL_EXECUTOR_H

#include "CuCore.h"
#include <string>
#include <stack>
#include "Math.h"
#include "Matrix.h"
#include "CudaUtils.h"
#include "oapCudaMatrixUtils.h"

#include "Logger.h"

#include "IKernelExecutor.h"

#define printCuError(cuResult)                                                 \
  if (cuResult != 0) {                                                         \
    const char* buffer;                                                        \
    cuGetErrorName(cuResult, &buffer);                                         \
    debug("\n%s %s : %d cuError: %s (%d)\n", __FUNCTION__, __FILE__, __LINE__, \
          buffer, cuResult);                                                   \
    abort();                                                                   \
  }

namespace oap
{
namespace cuda
{

void Init();

class CuDevice {
 public:
  CuDevice();
  virtual ~CuDevice();
  virtual void setDevice(CUdevice cuDecive) = 0;
  virtual void setDeviceInfo(const CuDevice& deviceInfo) = 0;
  virtual CUdevice getDevice() const = 0;
};

class CuDeviceInfo : public CuDevice {
  public:

    CuDeviceInfo();
    CuDeviceInfo(const CuDeviceInfo& orig);
    virtual ~CuDeviceInfo();

    CUdevice getDevice() const;

    void setDevice(CUdevice cuDecive);
    void setDeviceInfo(const CuDevice& deviceInfo);

    void getDeviceProperties (DeviceProperties& cuDevprop);

    uint getMaxThreadsX() const;
    uint getMaxThreadsY() const;
    uint getMaxBlocksX() const;
    uint getMaxBlocksY() const;

    uint getSharedMemorySize() const;
    uint getMaxThreadsPerBlock() const;

    void printDeviceInfo ()
    {
      initDeviceProperties ();

      logInfo ("Device properties: \n --Max grid size: %d, %d, %d.\n --Max threads dim: %d, %d, %d\
                --Max threads per block: %d \n --Register per block: %d \n --Shared memory per block in bytes: %d \n",
        m_deviceProperties.maxBlocksCount[0],
        m_deviceProperties.maxBlocksCount[1],
        m_deviceProperties.maxBlocksCount[2],
        m_deviceProperties.maxThreadsCount[0],
        m_deviceProperties.maxThreadsCount[1],
        m_deviceProperties.maxThreadsCount[2],
        m_deviceProperties.maxThreadsPerBlock,
        m_deviceProperties.regsPerBlock,
        m_deviceProperties.sharedMemPerBlock);
    }

    bool isInitialized () const
    {
      return m_initialized;
    }

  private:
    CUdevice m_cuDevice = 0;
  
    bool m_initialized = false;
  
    int m_values[9];
    const CUdevice_attribute m_attributes[9] =
    {
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, 
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    };

    void initDeviceProperties ();

  protected:
    DeviceProperties m_deviceProperties;
};

class Context : public CuDeviceInfo {
 public:
  static int FIRST;
  static int LAST;

  void create(int deviceIndex = -1);
  void destroy();
  static Context& Instance();

 protected:
  Context();
  virtual ~Context();

 private:
  static Context m_Context;
  std::stack<CUcontext> m_contexts;
  int deviceIndex;
};

class Kernel : public oap::IKernelExecutor
{
  public:
    Kernel();

    virtual ~Kernel();

    virtual std::string getErrorMsg () const override;

    void setDimensionsDevice(math::Matrix* dmatrix);

    virtual uint getMaxThreadsPerBlock() const override;

    bool load(const char* path);

    bool load(const char** pathes);

    void unload();

    void calculateThreadsBlocks(uint blocks[2], uint threads[2],
                              uint w, uint h);

    void calculateThreadsBlocksDevice(uint blocks[2], uint threads[2],
                                    math::Matrix* dmatrix);

    static void SetThreadsBlocks (uint blocks[2], uint threads[2], uint w, uint h, uint maxThreadsPerBlock);

    static bool Execute(const char* functionName, const void** params, oap::cuda::Kernel& kernel);

  protected:
    virtual bool run(const char* functionName) override;

  private:
    void* m_image;
    std::string m_path;
    CUmodule m_cuModule;

    void releaseImage();
    void unloadCuModule();
    void loadCuModule();
    void setImage(void* image);
};
}
}
#endif /* KERNELEXECUTOR_H */
