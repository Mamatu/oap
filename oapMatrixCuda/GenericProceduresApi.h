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

#ifndef OAP_GENERIC_PROCEDURES_API_H
#define OAP_GENERIC_PROCEDURES_API_H

#include <map>
#include <sstream>

#include "Buffer.h"

#include "Math.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "IKernelExecutor.h"

#include "GenericCoreApi.h"
#include "GenericValidationApi.h"

#include "oapHostMatrixUtils.h"

#define CHECK_MATRIX(m) debugAssertMsg (m != NULL, "Matrix is nullptr.");

namespace oap
{
namespace generic
{
  using SharedMemoryCallback = std::function<uintt(uint blocks[2], uint threads[2])>;

  struct Args
  {
    bool retrieveDims = true;
    uintt w;
    uintt h;

    bool prepareDims = true;
    uint blocks[2];
    uint threads[2];

    uintt sharedMemorySize = 0;
    SharedMemoryCallback smCallback = nullptr;
  };

  inline void prepareDims (oap::IKernelExecutor* kexec, uintt w, uintt h, uint blocks[2], uint threads[2])
  {
    kexec->calculateThreadsBlocks (blocks, threads, w, h);
    kexec->setBlocksCount (blocks[0], blocks[1]);
    kexec->setThreadsCount (threads[0], threads[1]);
  }

  template<typename PreExecCallback, typename PostExecCallback>
  bool execute (oap::IKernelExecutor* kexec, const char* functionName, uintt w, uintt h, void** params, uintt sharedMemory, bool _prepareDims,
               uint blocks[2], uint threads[2], PreExecCallback&& preExecCallback, PostExecCallback&& postExecCallback, const SharedMemoryCallback& smCallback = nullptr)
  {
    if (_prepareDims)
    {
      prepareDims (kexec, w, h, blocks, threads);
    }

    if (smCallback)
    {
      sharedMemory = smCallback (blocks, threads);
    }

    kexec->setSharedMemory (sharedMemory);

    preExecCallback ();

    kexec->setParams (params);
    bool status = kexec->execute(functionName);

    postExecCallback ();

    return status;
  }


  template<typename GetMatrixInfo, typename PreExecCallback>
  bool executeKernel (const std::string& kernelName, const math::MatrixInfo& ref, void** params, oap::IKernelExecutor* kexec, BasicMatrixApi<GetMatrixInfo>& bmApi, PreExecCallback&& preExecCallback, Args args = Args())
  {
    if (args.retrieveDims)
    {
      args.w = ref.columns ();
      args.h = ref.rows ();
    }

    return execute (kexec, kernelName.c_str(), args.w, args.h, params, args.sharedMemorySize, args.prepareDims, args.blocks, args.threads, preExecCallback, [](){}, args.smCallback);
  }

  using DefaultExecCallbackType = std::function<void()>;

  template<typename GetMatrixInfo, typename PreExecCallback = DefaultExecCallbackType, typename PostExecCallback = DefaultExecCallbackType>
  bool executeKernel (const std::string& kernelName, const math::MatrixInfo& ref, void** params, oap::IKernelExecutor* kexec, GetMatrixInfo&& getMatrixInfo,
                      Args args = Args(), PreExecCallback&& preExecCallback = [](){}, PostExecCallback&& postExecCallback = [](){})
  {
    if (args.retrieveDims)
    {
      args.w = ref.columns ();
      args.h = ref.rows ();
    }

    return execute (kexec, kernelName.c_str(), args.w, args.h, params, args.sharedMemorySize, args.prepareDims, args.blocks, args.threads, preExecCallback, postExecCallback, args.smCallback);
  }

  template<typename GetMatrixInfo, typename PreExecCallback = DefaultExecCallbackType, typename PostExecCallback = DefaultExecCallbackType>
  bool executeKernel (const std::string& kernelName, const math::Matrix* refMatrix, void** params, oap::IKernelExecutor* kexec, GetMatrixInfo&& getMatrixInfo,
                      Args args = Args(), PreExecCallback&& preExecCallback = [](){}, PostExecCallback&& postExecCallback = [](){})
  {
    auto ref = getMatrixInfo (refMatrix);

    return executeKernel (kernelName, ref, params, kexec, getMatrixInfo, args, preExecCallback, postExecCallback);
  }

  template<typename GetMatrixInfo, typename PreExecCallback>
  bool executeKernel (const std::string& kernelName, math::Matrix* ref, void** params, oap::IKernelExecutor* kexec, BasicMatrixApi<GetMatrixInfo>& bmApi, PreExecCallback&& preExecCallback, Args args = Args())
  {
    if (args.retrieveDims)
    {
      auto minfo = bmApi.getMatrixInfo (ref);

      args.w = minfo.columns ();
      args.h = minfo.rows ();
    }

    return execute (kexec, kernelName.c_str(), args.w, args.h, params, args.sharedMemorySize, args.prepareDims, args.blocks, args.threads, preExecCallback, [](){}, args.smCallback);
  }

  template<typename GetMatrixInfo, typename PreExecCallback>
  bool executeKernel1Arg (const std::string& kernelName, math::Matrix* output, const math::Matrix* arg, oap::IKernelExecutor* kexec, BasicMatrixApi<GetMatrixInfo>& bmApi, bool _prepareDims, PreExecCallback&& preExecCallback)
  {
    uint blocks[2];
    uint threads[2];

    auto minfo = bmApi.getMatrixInfo (output);

    const uintt w = minfo.columns ();
    const uintt h = minfo.rows ();

    void* params[] = {&output, &arg};

    return execute (kexec, kernelName.c_str(), w, h, params, 0, _prepareDims, blocks, threads, preExecCallback, [](){});
  }

  template<typename GetMatrixInfo, typename PreExecCallback, typename CreateKernelArray>
  bool executeKernel1Arg (const std::string& kernelName, math::Matrix* output, const math::Matrix* arg, uintt dims[2],
                          oap::IKernelExecutor* kexec, BasicMatrixApi<GetMatrixInfo>& bmApi, bool _prepareDims, PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    uint blocks[2];
    uint threads[2];

    auto minfo = bmApi.getMatrixInfo (output);

    const uintt w = minfo.columns ();
    const uintt h = minfo.rows ();

    uintt* karray = createKernelArray(dims, 2);
    void* params[] = {&output, &arg, &karray};

    return execute (kexec, kernelName.c_str(), w, h, params, 0, _prepareDims, blocks, threads, preExecCallback, [](){});
  }

  template<typename GetMatrixInfo, typename Copy, typename GetAddress>
  class SumApi : public BasicMatrixApi<GetMatrixInfo>
  {
    public:
      SumApi(GetMatrixInfo&& _getMatrixInfo, Copy&& _copy, GetAddress&& _getReAddress, GetAddress&& _getImAddress):
             BasicMatrixApi<GetMatrixInfo>(_getMatrixInfo), copy (_copy),
             getReAddress (_getReAddress), getImAddress (_getImAddress)
      {}

      Copy&& copy;
      GetAddress&& getReAddress;
      GetAddress&& getImAddress;
  };

  template<typename HBuffer, typename DBuffer>
  class SumBuffers
  {
    public:

      SumBuffers (HBuffer& tbh1, DBuffer& tbd1, HBuffer& tbh2, DBuffer& tbd2):
              re (std::pair<HBuffer&, DBuffer&>(tbh1, tbd1)),
              im (std::pair<HBuffer&, DBuffer&>(tbh2, tbd2))
      {}

      std::pair<HBuffer&, DBuffer&> re;
      std::pair<HBuffer&, DBuffer&> im;
  };

  template<typename GetMatrixInfo, typename Copy, typename GetAddress, typename HBuffer, typename DBuffer>
  bool sum (floatt& reoutput, floatt& imoutput, math::Matrix* matrix, oap::IKernelExecutor* kexec, SumApi<GetMatrixInfo, Copy, GetAddress>& sumApi, SumBuffers<HBuffer, DBuffer>& sumBuffers)
  {
    auto minfo = sumApi.getMatrixInfo (matrix);

    const uintt w = minfo.columns ();
    const uintt h = minfo.rows ();

    if (w * h == 1)
    {
      if (minfo.isRe)
      {
        sumApi.copy (&reoutput, sumApi.getReAddress (matrix), sizeof(floatt));
      }
      if (minfo.isIm)
      {
        sumApi.copy (&imoutput, sumApi.getImAddress (matrix), sizeof(floatt));
      }

      return true;
    }

    uint blocks[2];
    uint threads[2];

    kexec->calculateThreadsBlocks(blocks, threads, w, h);
    kexec->setBlocksCount(blocks[0], blocks[1]);
    kexec->setThreadsCount(threads[0], threads[1]);

    const uintt length = blocks[0] * blocks[1];

    if (minfo.isRe)
    {
      sumBuffers.re.second.realloc (length);
      sumBuffers.re.first.realloc (length);
    }

    if (minfo.isIm)
    {
      sumBuffers.im.second.realloc (length);
      sumBuffers.im.first.realloc (length);
    }

    void* params[] = {&sumBuffers.re.second.m_buffer, &sumBuffers.im.second.m_buffer, &matrix};

    uint factor = 1;
    if (minfo.isIm && minfo.isRe)
    {
      factor = 2;
    }

    bool cuStatus = kexec->execute ("CUDAKernel_SumShared", params, factor * threads[0] * threads[1] * sizeof(floatt));

    reoutput = 0;
    imoutput = 0;

    auto sum = [length](const HBuffer& buffer) -> floatt
    {
      floatt sum = 0;
      for (size_t idx = 0; idx < length; ++idx)
      {
        sum += buffer.m_buffer[idx];
      }
      return sum;
    };

    if (minfo.isRe)
    {
      sumApi.copy (sumBuffers.re.first.m_buffer, sumBuffers.re.second.m_buffer, sumBuffers.re.first.getSizeOfBuffer());
      reoutput = sum (sumBuffers.re.first);
    }

    if (minfo.isIm)
    {
      sumApi.copy (sumBuffers.im.first.m_buffer, sumBuffers.im.second.m_buffer, sumBuffers.im.first.getSizeOfBuffer());
      imoutput = sum (sumBuffers.im.first);
    }

    return cuStatus;
  }

  template<typename GetMatrixInfo>
  bool crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, oap::IKernelExecutor* kexec, BasicMatrixApi<GetMatrixInfo>& api)
  {
    auto minfo = api.getMatrixInfo (output);

    if (minfo.isIm)
    {
      debugAssert ("CrossEntropy doesn't support complex matrix (only real)");
      return false;
    }

    const uintt w = minfo.columns ();
    const uintt h = minfo.rows ();

    uint blocks[2];
    uint threads[2];

    kexec->calculateThreadsBlocks(blocks, threads, w, h);
    kexec->setBlocksCount(blocks[0], blocks[1]);
    kexec->setThreadsCount(threads[0], threads[1]);   
      
    void* params[] = {&output, &params0, &params1};

    return kexec->execute ("CUDAKernel_CrossEntropy", params);
  }

  template<typename BasicMatrixApi, typename PreExecCallback>
  bool dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                  uintt outputColumns, uintt outputRows,
                  oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                  PreExecCallback&& preExecCallback)
  {
    oap::generic::check_dotProduct (output, matrix1, matrix2, bmApi);

    const char* kname = "CUDAKernel_DotProduct";

    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = outputColumns;
    args.h = outputRows;

    args.prepareDims = true;

    void* params[] = {&output, &matrix1, &matrix2};

    return oap::generic::executeKernel (kname, output, params, kexec, bmApi, preExecCallback, args);
  }

  template<typename BasicMatrixApi, typename PreExecCallback>
  bool dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                  oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                  PreExecCallback&& preExecCallback)
  {
    auto outputInfo = bmApi.getMatrixInfo (output);
    uintt outputColumns = outputInfo.columns ();
    uintt outputRows = outputInfo.rows ();
  
    return dotProduct (output, matrix1, matrix2, outputColumns, outputRows, kexec, preExecCallback, bmApi);
  }

  template<typename BasicMatrixApi, typename PreExecCallback, typename CreateKernelArray>
  bool dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, uintt dims[3][2],
                  oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                  PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    auto oinfo = bmApi.getMatrixInfo (output);
    auto minfo1 = bmApi.getMatrixInfo (matrix1);
    auto minfo2 = bmApi.getMatrixInfo (matrix2);

    oap::generic::check_dotProduct (output, matrix1, matrix2, dims,
                                   oinfo, minfo1, minfo2);

    const char* kname = "CUDAKernel_DotProductDim";

    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = dims[0][0];
    args.h = dims[0][1];

    args.prepareDims = true;

    uintt hostEx[] = {args.w, args.h, dims[1][0]};
    uintt* kernelArray = createKernelArray (hostEx, sizeof(hostEx) / sizeof(uintt));
    void* params[] = {&output, &matrix1, &matrix2, &kernelArray};

    return oap::generic::executeKernel (kname, output, params, kexec, bmApi, preExecCallback, args);
  }

  template<typename BasicMatrixApi, typename PreExecCallback, typename CreateKernelArray>
  bool dotProductPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                          oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                          PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    auto oinfo = bmApi.getMatrixInfo (output);
    auto minfo1 = bmApi.getMatrixInfo (matrix1);
    auto minfo2 = bmApi.getMatrixInfo (matrix2);

    oap::generic::check_dotProductPeriodic (output, matrix1, matrix2, oinfo, minfo1, minfo2);

    void* params[] = {&output, &matrix1, &matrix2};
    const char* kname = "CUDAKernel_DotProductPeriodic";

    return oap::generic::executeKernel (kname, output, params, kexec, bmApi, preExecCallback);
  }

  template<typename BasicMatrixApi, typename PreExecCallback, typename CreateKernelArray>
  bool dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                              uintt dims[3][2], uintt periodicRows,
                              oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                              PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    auto oinfo = bmApi.getMatrixInfo (output);
    auto minfo1 = bmApi.getMatrixInfo (matrix1);
    auto minfo2 = bmApi.getMatrixInfo (matrix2);

    oap::generic::check_dotProductDimPeriodic (output, matrix1, matrix2, dims, periodicRows, oinfo, minfo1, minfo2);

    const char* kname = "CUDAKernel_DotProductDimPeriodic";

    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = dims[0][0];
    args.h = oinfo.rows();

    args.prepareDims = true;

    uintt hostEx[] = {dims[0][0], dims[0][1], dims[1][0], periodicRows};
    uintt* kernelArray = createKernelArray (hostEx, sizeof(hostEx) / sizeof(uintt));

    void* params[] = {&output, &matrix1, &matrix2, &kernelArray};

    return oap::generic::executeKernel (kname, output, params, kexec, bmApi, preExecCallback, args);
  }

  template<typename BasicMatrixApi, typename PreExecCallback, typename CreateKernelArray>
  bool tensorProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt dims[3][2],
                      oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                      PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    CHECK_MATRIX(output);
    CHECK_MATRIX(params0);
    CHECK_MATRIX(params1);

    auto oinfo = bmApi.getMatrixInfo (output);
    auto minfo0 = bmApi.getMatrixInfo (params0);
    auto minfo1 = bmApi.getMatrixInfo (params1);

    oap::generic::check_tensorProduct (output, params0, params1, dims, oinfo, minfo0, minfo1);

    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = dims[0][0];
    args.h = dims[0][1];

    uintt hostEx[] = {dims[0][0], dims[0][1], dims[1][0], dims[1][1], dims[2][0], dims[2][1]};
    uintt* kernelArray = createKernelArray (hostEx, sizeof(hostEx) / sizeof(uintt));

    void* params[] = {&output, &params0, &params1, &kernelArray};
    const char* kname = "CUDAKernel_TensorProductDim";

    return generic::executeKernel (kname, output, params, kexec, bmApi, preExecCallback, args);
  }

  template<typename Api, typename GetMatrixInfo, typename PreExecCallback>
  bool qrDecomposition_HT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT,
                          oap::IKernelExecutor* kexec, Api& api, GetMatrixInfo&& getMatrixInfo, PreExecCallback&& preExecCallback)
  {
    bool status = false;
    void* params[] = {&Q, &R, &A, &V, &VT, &P, &VVT};

    uint maxThreads = kexec->getMaxThreadsPerBlock();

    auto minfo = getMatrixInfo (A);
    const uintt w = minfo.columns();
    const uintt h = minfo.rows();

    const char* kname = "CUDAKernel_QRHT";
    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = w;
    args.h = h;

    args.smCallback = [](uint blocks[2], uint threads[2])
    {
      return sizeof(floatt) * threads[0] * threads[1];
    };

    if (maxThreads >= w * h)
    {
      status = generic::executeKernel (kname, minfo, params, kexec, getMatrixInfo, args, preExecCallback);
    }
    else
    {
      debugAssert ("not implemented yet" == nullptr);
      //m_cuStatus = HOSTKernel_QRGR(Q, R, H, aux0, aux1, aux2, aux3, m_kernel);
    }

    return status;
  }

  template<typename BasicMatrixApi, typename PreExecCallback>
  bool func (const std::string& kname, math::Matrix* output, math::Matrix* matrix,
                oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi, PreExecCallback&& preExecCallback)
  {
    math::MatrixInfo minfo1 = bmApi.getMatrixInfo (output);
    math::MatrixInfo minfo2 = bmApi.getMatrixInfo (matrix);

    check_isEqualDim (minfo1, minfo2);
 
    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = minfo1.columns();
    args.h = minfo1.rows();

    void* params[] = {&output, &matrix};

    return oap::generic::executeKernel (kname, minfo1, params, kexec, bmApi, preExecCallback, args);
  }

  template<typename BasicMatrixApi, typename PreExecCallback, typename CreateKernelArray>
  bool funcDim (const std::string& kname, math::Matrix* output, math::Matrix* matrix, uintt dims[2],
                oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    math::MatrixInfo minfo1 = bmApi.getMatrixInfo (output);
    math::MatrixInfo minfo2 = bmApi.getMatrixInfo (matrix);

    check_isEqualDim (minfo1, minfo2);
 
    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = dims[0];
    args.h = minfo1.rows();

    uintt hostEx[] = {dims[0], dims[1]};
    uintt* kernelArray = createKernelArray (hostEx, sizeof(hostEx) / sizeof(uintt));

    void* params[] = {&output, &matrix, &kernelArray};

    return oap::generic::executeKernel (kname, minfo1, params, kexec, bmApi, preExecCallback, args);
  }
  
  template<typename BasicMatrixApi, typename PreExecCallback, typename CreateKernelArray>
  bool funcDimPeriodic (const std::string& kname, math::Matrix* output, math::Matrix* matrix, uintt dims[2][2],
                       oap::IKernelExecutor* kexec, BasicMatrixApi& bmApi,
                       PreExecCallback&& preExecCallback, CreateKernelArray&& createKernelArray)
  {
    math::MatrixInfo minfo1 = bmApi.getMatrixInfo (output);
    math::MatrixInfo minfo2 = bmApi.getMatrixInfo (matrix);

    check_isEqualDim (minfo1, minfo2);
 
    oap::generic::Args args;

    args.retrieveDims = false;
    args.w = dims[0][0];
    args.h = minfo1.rows();

    uintt hostEx[] = {dims[0][0], dims[0][1], dims[1][1]};
    uintt* kernelArray = createKernelArray (hostEx, sizeof(hostEx) / sizeof(uintt));

    void* params[] = {&output, &matrix, &kernelArray};

    return oap::generic::executeKernel (kname, minfo1, params, kexec, bmApi, preExecCallback, args);
  }

  template<typename GetMatrixInfo, typename PreExecCallback>
  bool setIdentityMatrix (math::Matrix* matrix, oap::IKernelExecutor* kexec, GetMatrixInfo&& getMatrixInfo, PreExecCallback&& preExecCallback)
  {
    void* params[] = {&matrix};
    oap::generic::Args args;

    auto minfo = getMatrixInfo (matrix);

    const char* kname = "CUDAKernel_SetIdentity";

    args.retrieveDims = false;
    args.w = minfo.columns();
    args.h = minfo.rows();

    return executeKernel (kname, minfo, params, kexec, getMatrixInfo, args, preExecCallback, [](){});
  }
}
}

#endif
