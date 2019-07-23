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

#include <limits.h>
#include <linux/fs.h>
#include <cstdlib>

#include <math.h>

#include "Logger.h"
#include "KernelExecutor.h"
#include "ThreadsMapper.h"

#define printCuErrorStatus(status, cuResult)                                   \
  if (cuResult != 0) {                                                         \
    status = cuResult;                                                         \
    const char* buffer;                                                        \
    cuGetErrorName(cuResult, &buffer);                                         \
    logInfo("\n%s %s : %d cuError: %s (%d)\n", __FUNCTION__, __FILE__, __LINE__, \
          buffer, cuResult);                                                   \
    abort();                                                                   \
  }

namespace oap
{
namespace cuda
{

void PrintDeviceInfo(CUdevice cudevice) {
  CUdevprop cuDevprop;
  printCuError(cuDeviceGetProperties(&cuDevprop, cudevice));
  debug(
      "Device properties: \n --Max grid size: %d, %d, %d.\n --Max threads dim: "
      "%d, %d, %d",
      cuDevprop.maxGridSize[0], cuDevprop.maxGridSize[1],
      cuDevprop.maxGridSize[2], cuDevprop.maxThreadsDim[0],
      cuDevprop.maxThreadsDim[1], cuDevprop.maxThreadsDim[2]);
  debug(" --Max threads per block: %d", cuDevprop.maxThreadsPerBlock);
  debug(" --Register per block: %d", cuDevprop.regsPerBlock);
  debug(" --Shared memory per block in bytes: %d", cuDevprop.sharedMemPerBlock);
}

void Init() {
  static bool wasInit = false;
  if (wasInit == false) {
    wasInit = true;
    printCuError(cuInit(0));
  }
}

CuDevice::CuDevice() {}

CuDevice::~CuDevice() {}

CuDeviceInfo::CuDeviceInfo() : m_cuDevice(0) {}

CuDeviceInfo::CuDeviceInfo(const CuDeviceInfo& orig)
    : m_cuDevice(orig.m_cuDevice) {}

CuDeviceInfo::~CuDeviceInfo() { m_cuDevice = 0; }

CUdevice CuDeviceInfo::getDevice() const { return m_cuDevice; }

void CuDeviceInfo::getDeviceProperties(CUdevprop& cuDevprop) const {
  printCuError(cuDeviceGetProperties(&cuDevprop, m_cuDevice));
}

uint CuDeviceInfo::getMaxThreadsX() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxThreadsDim[0];
}

uint CuDeviceInfo::getMaxThreadsY() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxThreadsDim[1];
}

uint CuDeviceInfo::getMaxBlocksX() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxGridSize[0];
}

uint CuDeviceInfo::getMaxBlocksY() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxGridSize[1];
}

uint CuDeviceInfo::getSharedMemorySize() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.sharedMemPerBlock;
}

void CuDeviceInfo::setDevice(CUdevice cuDevice) {
  this->m_cuDevice = cuDevice;
}

void CuDeviceInfo::setDeviceInfo(const CuDevice& deviceInfo) {
  setDevice(deviceInfo.getDevice());
}

int Context::FIRST = 0;
int Context::LAST = INT_MAX;

Context::Context() {}

Context Context::m_Context;

Context& Context::Instance() { return Context::m_Context; }

void Context::create(int _deviceIndex) {

  Init();

  int count = -1;

  printCuError(cuDeviceGetCount(&count));
  debug("Devices count: %d \n", count);

  debugAssertMsg (count > 0, "No device detected. Count is equal 0!")

  deviceIndex = _deviceIndex;

  if (deviceIndex >= count)
  {
    deviceIndex = count - 1;
    debug("The last device will be used.");
  }

  if (deviceIndex < 0)
  {
    deviceIndex = 0;
    debug("The first device will be used.");
  }

  debugAssertMsg(deviceIndex >= 0 && deviceIndex < count, "Index of device is out of scope!");

  CUdevice device = 0;
  printCuError(cuDeviceGet(&device, deviceIndex));
  setDevice(device);
  CUcontext context;
  printCuError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
  m_contexts.push(context);
}

void Context::destroy() {
  if (!m_contexts.empty()) {
    CUcontext context = m_contexts.top();
    m_contexts.pop();
    printCuError(cuCtxDestroy(context));
  }
}

Context::~Context() { destroy(); }

void Kernel::setDimensionsDevice(math::Matrix* dmatrix) {
  uintt columns = CudaUtils::GetColumns(dmatrix);
  uintt rows = CudaUtils::GetRows(dmatrix);
  setDimensions(columns, rows);
}

Kernel::Kernel() : m_image(NULL), m_cuModule(NULL)
{
}

void Kernel::unloadCuModule() {
  if (m_cuModule != NULL) {
    cuModuleUnload(m_cuModule);
    m_cuModule = NULL;
  }
}

void Kernel::loadCuModule() {
  //unloadCuModule();
  if (NULL != m_image && NULL == m_cuModule) {
    debug("Load module from image = %p", m_image);
    printCuError(cuModuleLoadData(&m_cuModule, m_image));
  }
}

void Kernel::unload() { unloadCuModule(); }

void Kernel::setImage(void* image) {
  this->m_image = image;
  loadCuModule();
}

bool Kernel::run (const char* functionName)
{
  CUresult status = CUDA_SUCCESS;

  if (NULL == m_image && m_path.length() == 0)
  {
    debugError(
        "Error: image and path not defined. Function name: %s. Probalby was "
        "not executed load() method. \n",
        functionName);
  }

  loadCuModule();

  CUfunction cuFunction = NULL;
  if (NULL != m_cuModule)
  {
    printCuErrorStatus(
        status, cuModuleGetFunction(&cuFunction, m_cuModule, functionName));

    const uint* const threadsCount = getThreadsCount ();
    const uint* const blocksCount = getBlocksCount ();

#ifdef OAP_PRINT_KERNEL_INFO
    logInfo("Load kernel: %s", functionName);
    logInfo("Image: %p", m_image);
    logInfo("Module handle: %p", m_cuModule);
    logInfo("Function handle: %p", cuFunction);
    PrintDeviceInfo(getDevice());
    logInfo(" Execution:");
    logInfo(" --threads counts: %d, %d, %d", threadsCount[0], threadsCount[1], threadsCount[2]);
    logInfo(" --blocks counts: %d, %d, %d", blocksCount[0], blocksCount[1], blocksCount[2]);
    logInfo(" --shared memory in bytes: %d", getSharedMemory());
#endif
    printCuErrorStatus(status,
        cuLaunchKernel(cuFunction,
                       blocksCount[0], blocksCount[1], blocksCount[2],
                       threadsCount[0], threadsCount[1], threadsCount[2],
                       getSharedMemory(), NULL,
                       this->getParams(), NULL));
  }
  else
  {
    if (NULL != m_image) {
      debug("Module is incorrect %p %p;", m_cuModule, m_image);
    } else if (m_path.length() > 0) {
      debug("Module is incorrect %p %s;", m_cuModule, m_path.c_str());
    }
    abort();
  }

  //unloadCuModule();

  return status == 0;
}

Kernel::~Kernel() {
  unloadCuModule();
  releaseImage();
}

std::string Kernel::getErrorMsg () const
{
  return "";
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

typedef std::vector<std::string> Strings;

void split(Strings& pathes, const std::string& env, char c) {
  size_t pindex = 0;
  size_t index = 0;
  while ((index = env.find(c, pindex)) != std::string::npos) {
    pathes.push_back(env.substr(pindex, index - pindex));
  }
  pathes.push_back(env.substr(pindex, env.length() - pindex));
}

void getSysPathes(Strings& pathes) {
  char* cenvs = ::getenv("OAP_CUBIN_PATH");
  if (cenvs != NULL) {
    std::string env = cenvs;
    split(pathes, env, ':');
  }
}

inline char* readData(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f != NULL) {
    char* data = loadData(f);
    return data;
  }
  return NULL;
}

inline char* readData(const char* path, const Strings& sysPathes) {
  for (size_t fa = 0; fa < sysPathes.size(); ++fa) {
    std::string p = sysPathes[fa] + "/" + path;
    char* data = readData(p.c_str());
    if (data != NULL) {
      return data;
    }
  }
  return NULL;
}

inline char* loadData(std::string& loadedPath, const char** pathes,
                      bool extraSysPathes = true) {
  std::vector<std::string> sysPathes;
  if (extraSysPathes) {
    getSysPathes(sysPathes);
  }
  while (pathes != NULL && *pathes != NULL) {
    char* data = readData(*pathes);
    if (data != NULL) {
      loadedPath = *pathes;
      return data;
    }
    if (extraSysPathes) {
      data = readData(*pathes, sysPathes);
      if (data != NULL) {
        loadedPath = *pathes;
        return data;
      }
    }
    ++pathes;
  }
  return NULL;
}

inline char* loadData(const char* path, bool extraSysPathes = true) {
  const char* pathes[] = {path, NULL};
  std::string lpath;
  return loadData(lpath, pathes, extraSysPathes);
}

bool Kernel::load(const char* path) {
  setImage(loadData(path));
#ifdef DEBUG
  if (m_image == NULL) {
    debug("Cannot load %s.\n", path);
  } else {
    debug("Loaded %s.\n", path);
  }
#endif
  return m_image != NULL;
}

bool Kernel::load(const char** pathes) {
  std::string path;
  setImage(loadData(path, pathes));
#ifdef DEBUG
  if (m_image == NULL) {
    debug("Cannot load %s.\n", path.c_str());
  } else {
    debug("Loaded %s.\n", path.c_str());
  }
#endif
  return m_image != NULL;
}

void Kernel::releaseImage() {
  if (m_image != NULL) {
    char* data = (char*)m_image;
    delete[] data;
  }
}

void Kernel::calculateThreadsBlocks(uint blocks[2], uint threads[2],
                                    uint w, uint h)
{
  SetThreadsBlocks(blocks, threads, w, h, getMaxThreadsPerBlock());
}

void Kernel::calculateThreadsBlocksDevice(uint blocks[2], uint threads[2],
                                          math::Matrix* dmatrix) {
  uintt columns = CudaUtils::GetColumns(dmatrix);
  uintt rows = CudaUtils::GetRows(dmatrix);
  calculateThreadsBlocks(blocks, threads, columns, rows);
}

void Kernel::SetThreadsBlocks(uint blocks[2], uint threads[2],
                              uint w, uint h,
                              uint maxThreadsPerBlock)
{
  utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
}

bool Kernel::Execute(const char* functionName, void** params, oap::cuda::Kernel& kernel)
{
  kernel.setParams(params);
  return kernel.execute(functionName);
}

uint Kernel::getMaxThreadsPerBlock() const
{
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxThreadsPerBlock;
}

}
}
