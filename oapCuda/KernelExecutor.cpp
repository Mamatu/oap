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



#include <linux/fs.h>
#include <math.h>

#include "KernelExecutor.h"
#include "ArrayTools.h"
#include "DebugLogs.h"
#include "ThreadsMapper.h"
#include <cstdlib>

//#define KERNEL_EXECUTOR_LOGS_FUNCTION_NAME
#define KERNEL_EXTENDED_INFO 0

namespace device {

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

DefaultDeviceInfo::DefaultDeviceInfo() : m_cuDevice(0) {}

DefaultDeviceInfo::DefaultDeviceInfo(const DefaultDeviceInfo& orig)
    : m_cuDevice(orig.m_cuDevice) {}

DefaultDeviceInfo::~DefaultDeviceInfo() { m_cuDevice = 0; }

CUdevice DefaultDeviceInfo::getDevice() const { return m_cuDevice; }

void DefaultDeviceInfo::getDeviceProperties(CUdevprop& cuDevprop) const {
  printCuError(cuDeviceGetProperties(&cuDevprop, m_cuDevice));
}

uint DefaultDeviceInfo::getMaxThreadsPerBlock() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxThreadsPerBlock;
}

uint DefaultDeviceInfo::getMaxThreadsX() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxThreadsDim[0];
}

uint DefaultDeviceInfo::getMaxThreadsY() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxThreadsDim[1];
}

uint DefaultDeviceInfo::getMaxBlocksX() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxGridSize[0];
}

uint DefaultDeviceInfo::getMaxBlocksY() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.maxGridSize[1];
}

uint DefaultDeviceInfo::getSharedMemorySize() const {
  CUdevprop cuDevprop;
  getDeviceProperties(cuDevprop);
  return cuDevprop.sharedMemPerBlock;
}

void DefaultDeviceInfo::setDevice(CUdevice cuDecive) {
  this->m_cuDevice = m_cuDevice;
}

void DefaultDeviceInfo::setDeviceInfo(const CuDevice& deviceInfo) {
  setDevice(deviceInfo.getDevice());
}

Context::Context(int _deviceIndex) : deviceIndex(_deviceIndex) {}

Context Context::m_Context;

Context& Context::Instance() { return Context::m_Context; }

void Context::create() {
  Init();
  int count = 0;
  printCuError(cuDeviceGetCount(&count));
  debug("Devices count: %d \n", count);
  deviceIndex = count - 1;
  if (deviceIndex < count) {
    CUdevice device = 0;
    printCuError(cuDeviceGet(&device, deviceIndex));
    setDevice(device);
    CUcontext context;
    printCuError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
    m_contexts.push(context);
  }
}

void Context::destroy() {
  if (!m_contexts.empty()) {
    CUcontext context = m_contexts.top();
    m_contexts.pop();
    printCuError(cuCtxDestroy(context));
  }
}

Context::~Context() { destroy(); }

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

uint Kernel::getThreadsX() const { return m_threadsCount[0]; }

uint Kernel::getThreadsY() const { return m_threadsCount[1]; }

uint Kernel::getBlocksX() const { return m_blocksCount[0]; }

uint Kernel::getBlocksY() const { return m_blocksCount[1]; }

void Kernel::setDimensions(uintt w, uintt h) {
  CUdevprop devprop;
  getDeviceProperties(devprop);
  calculateThreadsBlocks(m_blocksCount, m_threadsCount, w, h);
}

void Kernel::setDimensionsDevice(math::Matrix* dmatrix) {
  uintt columns = CudaUtils::GetColumns(dmatrix);
  uintt rows = CudaUtils::GetRows(dmatrix);
  setDimensions(columns, rows);
}

void Kernel::setSharedMemory(uintt sizeInBytes) {
  m_sharedMemoryInBytes = sizeInBytes;
}

void Kernel::setParams(void** params) { m_params = params; }

Kernel::Kernel()
    : m_params(NULL), m_paramsSize(0), m_image(NULL), m_cuModule(NULL) {
  this->m_blocksCount[0] = 1;
  this->m_blocksCount[1] = 1;
  this->m_blocksCount[2] = 1;
  this->m_threadsCount[0] = 1;
  this->m_threadsCount[1] = 1;
  this->m_threadsCount[2] = 1;
  m_sharedMemoryInBytes = 0;
}

int Kernel::getParamsCount() const { return this->m_paramsSize; }

void** Kernel::getParams() const { return this->m_params; }

void Kernel::unloadCuModule() {
  if (m_cuModule != NULL) {
    cuModuleUnload(m_cuModule);
    m_cuModule = NULL;
  }
}

void Kernel::loadCuModule() {
  unloadCuModule();
  if (NULL != m_image) {
    printCuError(cuModuleLoadData(&m_cuModule, m_image));
  }
}

void Kernel::unload() { unloadCuModule(); }

void Kernel::setImage(void* image) {
  this->m_image = image;
  // loadCuModule();
}

CUresult Kernel::execute(const char* functionName) {
  CUresult status = CUDA_SUCCESS;
  if (NULL == m_image && m_path.length() == 0) {
    debugError(
        "Error: image and path not defined. Function name: %s. Probalby was "
        "not executed load() method. \n",
        functionName);
  }
  loadCuModule();
  CUfunction cuFunction = NULL;
  if (NULL != m_cuModule) {
    printCuErrorStatus(
        status, cuModuleGetFunction(&cuFunction, m_cuModule, functionName));
#if KERNEL_EXTENDED_INFO == 1
    debug("Load kernel: %s", functionName);
    debug("Image: %p", m_image);
    debug("Module handle: %p", m_cuModule);
    debug("Function handle: %p", cuFunction);
    PrintDeviceInfo(getDevice());
    debug(" Execution:");
    debug(" --threads counts: %d, %d, %d", m_threadsCount[0], m_threadsCount[1],
          m_threadsCount[2]);
    debug(" --blocks counts: %d, %d, %d", m_blocksCount[0], m_blocksCount[1],
          m_blocksCount[2]);
    debug(" --shared memory in bytes: %d", m_sharedMemoryInBytes);
#endif
    printCuErrorStatus(
        status,
        cuLaunchKernel(cuFunction, m_blocksCount[0], m_blocksCount[1],
                       m_blocksCount[2], m_threadsCount[0], m_threadsCount[1],
                       m_threadsCount[2], m_sharedMemoryInBytes, NULL,
                       this->getParams(), NULL));
  } else {
    if (NULL != m_image) {
      debug("Module is incorrect %p %p;", m_cuModule, m_image);
    } else if (m_path.length() > 0) {
      debug("Module is incorrect %d %s;", m_cuModule, m_path.c_str());
    }
    abort();
  }
  unloadCuModule();
  resetParameters();
  return status;
}

Kernel::~Kernel() {
  unloadCuModule();
  releaseImage();
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
  char* cenvs = ::getenv("CUBIN_PATHES");
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

void* Kernel::LoadImage(const char* path) { return loadData(path, true); }

void* Kernel::LoadImage(std::string& path, const char** pathes) {
  return loadData(path, pathes, true);
}

void Kernel::FreeImage(void* image) {
  char* data = reinterpret_cast<char*>(image);
  delete[] data;
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

void Kernel::resetParameters() {
  this->m_blocksCount[0] = 1;
  this->m_blocksCount[1] = 1;
  this->m_blocksCount[2] = 1;
  this->m_threadsCount[0] = 1;
  this->m_threadsCount[1] = 1;
  this->m_threadsCount[2] = 1;
  m_sharedMemoryInBytes = 0;
}

void Kernel::calculateThreadsBlocks(uintt blocks[2], uintt threads[2], uintt w,
                                    uintt h) {
  SetThreadsBlocks(blocks, threads, w, h, getMaxThreadsPerBlock());
}

void Kernel::calculateThreadsBlocksDevice(uintt blocks[2], uintt threads[2],
                                          math::Matrix* dmatrix) {
  uintt columns = CudaUtils::GetColumns(dmatrix);
  uintt rows = CudaUtils::GetRows(dmatrix);
  calculateThreadsBlocks(blocks, threads, columns, rows);
}

void Kernel::SetThreadsBlocks(uintt blocks[2], uintt threads[2], uintt w,
                              uintt h, uintt maxThreadsPerBlock) {
  utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
}

CUresult Kernel::Execute(const char* functionName, void** params,
                         ::device::Kernel& kernel) {
  kernel.setParams(params);
  return kernel.execute(functionName);
}
}
