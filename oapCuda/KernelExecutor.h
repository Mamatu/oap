/*
 * Copyright 2016 - 2019 Marcin Matula
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
  CUdevice m_cuDevice;

 public:

  CuDeviceInfo();
  CuDeviceInfo(const CuDeviceInfo& orig);
  virtual ~CuDeviceInfo();

  CUdevice getDevice() const;

  void setDevice(CUdevice cuDecive);
  void setDeviceInfo(const CuDevice& deviceInfo);

  void getDeviceProperties(CUdevprop& cuDevprop) const;

  uint getMaxThreadsX() const;
  uint getMaxThreadsY() const;
  uint getMaxBlocksX() const;
  uint getMaxBlocksY() const;

  uint getSharedMemorySize() const;
};

class Context : public CuDeviceInfo {
 public:
  static int FIRST;
  static int LAST;

  void create(int deviceIndex = Context::LAST);
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

class Kernel : public oap::IKernelExecutor, public CuDeviceInfo {
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

    static bool Execute(const char* functionName, void** params, oap::cuda::Kernel& kernel);

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
