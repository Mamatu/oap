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

#include <limits.h>
#include <cstdlib>

#include <math.h>
#include <cstring>

#include "Logger.hpp"
#include "KernelExecutor.hpp"
#include "ThreadsMapper.hpp"
#include "Config.hpp"

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

void Init()
{
  static bool wasInit = false;
  if (wasInit == false)
  {
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

void CuDeviceInfo::getDeviceProperties (DeviceProperties& devProp)
{
  initDeviceProperties ();

  std::memcpy (&devProp, &m_deviceProperties, sizeof (DeviceProperties));
}

uint CuDeviceInfo::getMaxThreadsX() const
{
  debugAssert (m_initialized);
  return m_deviceProperties.maxThreadsCount[0];
}

uint CuDeviceInfo::getMaxThreadsY() const
{
  debugAssert (m_initialized);
  return m_deviceProperties.maxThreadsCount[1];
}

uint CuDeviceInfo::getMaxBlocksX() const
{
  debugAssert (m_initialized);
  return m_deviceProperties.maxBlocksCount[0];
}

uint CuDeviceInfo::getMaxBlocksY() const
{
  debugAssert (m_initialized);
  return m_deviceProperties.maxBlocksCount[1];
}

uint CuDeviceInfo::getSharedMemorySize() const
{
  debugAssert (m_initialized);
  return m_deviceProperties.sharedMemPerBlock;
}

uint CuDeviceInfo::getMaxThreadsPerBlock() const
{
  debugAssert (m_initialized);
  return m_deviceProperties.maxThreadsPerBlock;
}

void CuDeviceInfo::setDevice (CUdevice cuDevice)
{
  this->m_cuDevice = cuDevice;
  initDeviceProperties ();
}

void CuDeviceInfo::setDeviceInfo (const CuDevice& deviceInfo)
{
  setDevice(deviceInfo.getDevice());
}

void CuDeviceInfo::initDeviceProperties ()
{
  if (!m_initialized)
  {
    for (size_t idx = 0; idx < 9; ++idx)
    {
      int result;
      printCuError (cuDeviceGetAttribute (&result, m_attributes[idx], m_cuDevice));
      m_values[idx] = result;
    }

    std::memcpy (&m_deviceProperties, m_values, sizeof(m_values));
    m_initialized = true;
  }
}

int Context::FIRST = 0;
int Context::LAST = INT_MAX;

Context::Context() {}

Context Context::m_Context;

Context& Context::Instance() { return Context::m_Context; }

void Context::create (int _deviceIndex)
{
  Init();

  int count = -1;

  printCuError(cuDeviceGetCount(&count));
  logDebug("Devices count: %d \n", count);

  debugAssertMsg (count > 0, "No device detected. Count is equal 0!")

  deviceIndex = _deviceIndex;

  if (deviceIndex >= count)
  {
    deviceIndex = count - 1;
    logDebug("The last device (%d) will be used.",deviceIndex);
  }
  else if (deviceIndex < 0)
  {
    std::string varval = oap::utils::Config::getVariable ("OAP_CU_DEVICE_INDEX");

    if (varval.empty())
    {
      deviceIndex = 0;
      logDebug ("The first device will be used.");
    }
    else
    {
      deviceIndex = std::stoi(varval.c_str());
      logDebug ("The device with number %d will be used.", deviceIndex);
    }
  }

  debugAssertMsg(deviceIndex >= 0 && deviceIndex < count, "Index of device is out of scope!");

  CUdevice device = 0;

  printCuError(cuDeviceGet(&device, deviceIndex));
  setDevice (device);

  CUcontext context;
  printCuError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));

  m_contexts.push(context);
}

void Context::destroy()
{
  if (!m_contexts.empty())
  {
    CUcontext context = m_contexts.top();

    m_contexts.pop();

    printCuError(cuCtxDestroy(context));
  }
}

Context::~Context() { destroy(); }

void Kernel::setDimensionsDevice(math::ComplexMatrix* dmatrix) {
  uintt columns = oap::cuda::GetColumns(dmatrix);
  uintt rows = oap::cuda::GetRows(dmatrix);
  setDimensions (columns, rows);
}

Kernel::Kernel() : m_image(NULL), m_cuModule(NULL)
{
}

void Kernel::loadCuModule() {
  //unloadCuModule();
  if (NULL != m_image && NULL == m_cuModule)
  {
    debug("Load module from image = %p", m_image);
    printCuError(cuModuleLoadData(&m_cuModule, m_image));
  }
}

void Kernel::unloadCuModule()
{
  if (m_cuModule != NULL)
  {
    cuModuleUnload(m_cuModule);
    m_cuModule = NULL;
  }
}

void Kernel::unload() { unloadCuModule(); }

void Kernel::setImage(void* image) {
  this->m_image = image;
  logInfo ("image = %p", m_image);
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

    const ExecutionParams& ep = getExecutionParams ();
#ifdef OAP_PRINT_KERNEL_INFO
    logInfo("Load kernel: %s", functionName);
    logInfo("Image: %p", m_image);
    logInfo("Module handle: %p", m_cuModule);
    logInfo("Function handle: %p", cuFunction);
    logInfo(" Execution:");
    logInfo(" --threads counts: %d, %d, %d", ep.threadsCount[0], ep.threadsCount[1], ep.threadsCount[2]);
    logInfo(" --blocks counts: %d, %d, %d", ep.blocksCount[0], ep.blocksCount[1], ep.blocksCount[2]);
    logInfo(" --shared memory in bytes: %d", ep.sharedMemSize);
#endif
    printCuErrorStatus(status,
        cuLaunchKernel(cuFunction,
                       ep.blocksCount[0], ep.blocksCount[1], ep.blocksCount[2],
                       ep.threadsCount[0], ep.threadsCount[1], ep.threadsCount[2],
                       ep.sharedMemSize, NULL,
                       const_cast<void**>(this->getParams()), NULL));
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

  return status == 0;
}

Kernel::~Kernel()
{
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
    logInfo ("%s", p.c_str());
    char* data = readData(p.c_str());
    if (data != NULL) {
      return data;
    }
  }
  return NULL;
}

inline char* loadData(std::string& loadedPath, const char** pathes, bool extraSysPathes = true)
{
  std::vector<std::string> sysPathes;
  if (extraSysPathes)
  {
    getSysPathes(sysPathes);
  }
  while (pathes != NULL && *pathes != NULL)
  {
    char* data = readData(*pathes);
    if (data != NULL)
    {
      loadedPath = *pathes;
      logInfo ("Loaded from %s", loadedPath.c_str());
      return data;
    }
    else
    {
      logInfo ("Cannot load from %s", *pathes);
    }
    if (extraSysPathes)
    {
      data = readData(*pathes, sysPathes);
      if (data != NULL)
      {
        loadedPath = *pathes;
        logInfo ("Loaded from %s", loadedPath.c_str());
        return data;
      }
      else
      {
        logInfo ("Cannot load from %s", *pathes);
      }
    }
    ++pathes;
  }
  return NULL;
}

inline char* loadData(const char* path, bool extraSysPathes = true) {
  const char* pathes[] = {path, NULL};
  std::string lpath;
  char* data = loadData(lpath, pathes, extraSysPathes);
#ifdef DEBUG
  debug ("Loaded: %p %s", data, lpath.c_str());
#endif
  return data;
}

bool Kernel::load(const char* path)
{
  setImage(loadData(path));
  if (m_image == NULL)
  {
    logInfo("Cannot load %s.", path);
  }
  else
  {
    logInfo("Loaded %s.", path);
  }
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
    logError ("~image = %p", m_image);
    char* data = (char*)m_image;
    delete[] data;
    m_image = NULL;
  }
}

void Kernel::calculateThreadsBlocks(uint blocks[2], uint threads[2],
                                    uint w, uint h)
{
  SetThreadsBlocks(blocks, threads, w, h, getMaxThreadsPerBlock());
}

void Kernel::calculateThreadsBlocksDevice(uint blocks[2], uint threads[2],
                                          math::ComplexMatrix* dmatrix) {
  uintt columns = oap::cuda::GetColumns(dmatrix);
  uintt rows = oap::cuda::GetRows(dmatrix);
  calculateThreadsBlocks(blocks, threads, columns, rows);
}

void Kernel::SetThreadsBlocks(uint blocks[2], uint threads[2],
                              uint w, uint h,
                              uint maxThreadsPerBlock)
{
  oap::utils::mapper::SetThreadsBlocks(blocks, threads, w, h, maxThreadsPerBlock);
}

bool Kernel::Execute(const char* functionName, const void** params, oap::cuda::Kernel& kernel)
{
  kernel.setParams(params);
  return kernel.execute(functionName);
}

uint Kernel::getMaxThreadsPerBlock() const
{
  return Context::Instance().getMaxThreadsPerBlock ();
}

}
}
