/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "PngFile.h"
#include "IEigenCalculator.h"
#include "DeviceDataLoader.h"
#include "Exceptions.h"
#include "MatchersUtils.h"

#include "ArnoldiProceduresImpl.h"
#include "oapCudaMatrixUtils.h"
#include "CuProceduresApi.h"
#include "oapDeviceMatrixUPtr.h"
#include "oapHostMatrixUPtr.h"

#include <memory>

using namespace ::testing;

class EigenCalculator : public oap::IEigenCalculator
{
  public:
    EigenCalculator (CuHArnoldiCallback* cuhArnoldi) : oap::IEigenCalculator(cuhArnoldi) {}

    ~EigenCalculator() {}

    void setEigenvaluesOutput(floatt* eigenvalues)
    {
      oap::IEigenCalculator::setEigenvaluesOutput (eigenvalues);
    }

    void setEigenvectorsOutput(math::Matrix** eigenvecs, ArnUtils::Type type)
    {
      oap::IEigenCalculator::setEigenvectorsOutput (eigenvecs, type);
    }

    oap::DeviceDataLoader* getDataLoader() const
    {
      return oap::IEigenCalculator::getDataLoader ();
    }
};

class ArnoldiOperations {
 public:
  oap::DeviceDataLoader* m_dataLoader;
  math::Matrix* value;
  oap::CuProceduresApi cuProceduresApi;

  ArnoldiOperations(oap::DeviceDataLoader* dataLoader)
      : m_dataLoader(dataLoader) {
    value = oap::cuda::NewDeviceReMatrix(1, 1);
  }

  ~ArnoldiOperations() { oap::cuda::DeleteDeviceMatrix(value); }

  static void multiplyFunc(math::Matrix* m_w, math::Matrix* m_v,
                           oap::CuProceduresApi& cuProceduresApi,
                           void* userData, CuHArnoldi::MultiplicationType mt)
  {
    if (mt == CuHArnoldi::TYPE_WV) {
      ArnoldiOperations* ao = static_cast<ArnoldiOperations*>(userData);
      oap::DeviceDataLoader* dataLoader = ao->m_dataLoader;

      math::MatrixInfo matrixInfo = dataLoader->getMatrixInfo();

      for (uintt index = 0; index < matrixInfo.m_matrixDim.columns; ++index) {
        math::Matrix* vec = dataLoader->createDeviceRowVector(index);

        //oap::cuda::PrintMatrix("vec =", vec);

        cuProceduresApi.dotProduct(ao->value, vec, m_v);
        oap::cuda::SetMatrix(m_w, ao->value, 0, index);

        oap::cuda::DeleteDeviceMatrix(vec);
      }
    }
  }

  bool verifyOutput (math::Matrix* vector, floatt value, EigenCalculator* eigenCalc)
  {
    math::Matrix* matrix = m_dataLoader->createMatrix();

    const uintt partSize = matrix->columns;

    math::Matrix* dvector = NULL;
    uintt vectorrows = 0;

    bool dvectorIsCopy = false;

    vectorrows = matrix->columns;

    if (eigenCalc->getEigenvectorsType() == ArnUtils::HOST) {
      dvector = oap::cuda::NewDeviceMatrixCopy(vector);
      dvectorIsCopy = true;
    } else if (eigenCalc->getEigenvectorsType() == ArnUtils::DEVICE) {
      dvector = vector;
      dvectorIsCopy = false;
    }

    if (dvector == NULL)  {
      debugAssert("Invalid eigenvectors type.");
    }

    oap::HostMatrixUPtr refMatrix = oap::host::NewMatrix(matrix, matrix->columns, partSize);

    oap::host::CopyMatrix(refMatrix, matrix);

    oap::DeviceMatrixUPtr drefMatrix = oap::cuda::NewDeviceMatrixCopy(refMatrix);

    math::MatrixInfo info = oap::host::GetMatrixInfo(refMatrix);

    const uintt matrixcolumns = info.m_matrixDim.columns;
    uintt matrixrows = info.m_matrixDim.rows;

    matrixrows = partSize;

    oap::DeviceMatrixUPtr matrix1 = oap::cuda::NewDeviceReMatrix(matrixrows, matrixcolumns);

    oap::DeviceMatrixUPtr leftMatrix = oap::cuda::NewDeviceReMatrix(matrixrows, matrixrows);
    oap::DeviceMatrixUPtr rightMatrix = oap::cuda::NewDeviceReMatrix(matrixrows, matrixrows);

    oap::DeviceMatrixUPtr vectorT = oap::cuda::NewDeviceReMatrix(vectorrows, 1);

    cuProceduresApi.transpose(matrix1, drefMatrix);
    cuProceduresApi.transpose(vectorT, dvector);
    cuProceduresApi.dotProduct(leftMatrix, drefMatrix, matrix1);

    floatt value2 = value * value;
    cuProceduresApi.multiplyReConstant(vectorT, vectorT, value2);
    cuProceduresApi.dotProduct(rightMatrix, dvector, vectorT);
    bool compareResult = cuProceduresApi.compare(leftMatrix, rightMatrix);

    oap::HostMatrixUPtr hleftMatrix = oap::host::NewReMatrix(CudaUtils::GetColumns(leftMatrix), CudaUtils::GetRows(leftMatrix));
    oap::HostMatrixUPtr hrightMatrix = oap::host::NewReMatrix(CudaUtils::GetColumns(rightMatrix), CudaUtils::GetRows(rightMatrix));

    oap::cuda::CopyDeviceMatrixToHostMatrix(hrightMatrix, rightMatrix);
    oap::cuda::CopyDeviceMatrixToHostMatrix(hleftMatrix, leftMatrix);

    EXPECT_THAT(hleftMatrix.get(), MatrixIsEqual(hrightMatrix.get(), InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));

    oap::host::DeleteMatrix(matrix);
    if (dvectorIsCopy) {
      oap::cuda::DeleteDeviceMatrix(dvector);
    }

    return compareResult;
  }

  math::Matrix* createDeviceMatrix() const {
    return m_dataLoader->createDeviceMatrix();
  }
};

class OapEigenCalculatorTests : public testing::Test {
 public:
  virtual void SetUp() { oap::cuda::Context::Instance().create(); }

  virtual void TearDown() { oap::cuda::Context::Instance().destroy(); }
};

class MatricesDeleter {
  int m_eigensCount;
  ArnUtils::Type m_type;
  public:
    MatricesDeleter(int eigensCount, ArnUtils::Type type) :
      m_eigensCount(eigensCount), m_type(type) {}

    MatricesDeleter& operator() (math::Matrix** evectors) {
      for (int fa = 0; fa < m_eigensCount; ++fa) {
        debug("Deleted matrix %p", evectors[fa]);
        if (m_type == ArnUtils::HOST) {
          oap::host::DeleteMatrix(evectors[fa]);
        } else if (m_type == ArnUtils::DEVICE) {
          oap::cuda::DeleteDeviceMatrix(evectors[fa]);
        }
      }
      delete[] evectors;
      return *this;
    }

};

using MatricesUPtr = std::unique_ptr<math::Matrix*, MatricesDeleter>;

class TestCuHArnoldiCallback : public CuHArnoldiCallback {
 public:
  TestCuHArnoldiCallback(ArnoldiOperations* ao, int counterLimit = 5) : m_ao(ao), m_counterLimit(counterLimit), m_counter(0) {
  }

  bool checkEigenspair(floatt revalue, floatt imvalue, math::Matrix* vector, uint index, uint max) {
    ++m_counter;
    debug("counter = %d", m_counter);

    bool output = (m_counter < m_counterLimit);
    if (output == false && index == max - 1) {
      m_counter = 0;
    }
    return output;
  }

  static MatricesUPtr launchTest(ArnUtils::Type eigensType, const oap::DataLoader::Info& info,
                                 int wantedEigensCount, int maxIterationCounter = 5)
  {
    std::unique_ptr<oap::DeviceDataLoader> dataLoader(
        oap::DeviceDataLoader::createDataLoader<oap::PngFile, oap::DeviceDataLoader>(info));

    ArnoldiOperations ao(dataLoader.get());
    TestCuHArnoldiCallback cuharnoldi(&ao, maxIterationCounter);
    cuharnoldi.setCallback(ArnoldiOperations::multiplyFunc, &ao);

    floatt reoevalues[wantedEigensCount];

    EigenCalculator eigenCalculator(&cuharnoldi);

    eigenCalculator.setDataLoader (dataLoader.get());

    eigenCalculator.setEigensCount(wantedEigensCount, wantedEigensCount);

    math::MatrixInfo matrixInfo = eigenCalculator.getMatrixInfo();

    MatricesDeleter matricesDeleter(wantedEigensCount, eigensType);

    MatricesUPtr evectorsUPtr(new math::Matrix* [wantedEigensCount], matricesDeleter);

    auto matricesInitializer = [&evectorsUPtr, wantedEigensCount, &matrixInfo, eigensType]() {
      math::Matrix** evectors = evectorsUPtr.get();
      const uintt rows = matrixInfo.m_matrixDim.rows;
      for (int fa = 0; fa < wantedEigensCount; ++fa) {
        if (eigensType == ArnUtils::HOST) {
          evectors[fa] = oap::host::NewReMatrix(1, rows);
        } else if (eigensType == ArnUtils::DEVICE) {
          evectors[fa] = oap::cuda::NewDeviceReMatrix(1, rows);
        }
        debug("Created matrix %p", evectors[fa]);
      }
    };

    matricesInitializer();

    math::Matrix** evectors = evectorsUPtr.get();

    eigenCalculator.setEigenvaluesOutput(reoevalues);

    eigenCalculator.setEigenvectorsOutput(evectors, eigensType);

    eigenCalculator.calculate();

    for (int fa = 0; fa < wantedEigensCount; ++fa) {
      EXPECT_TRUE(ao.verifyOutput(evectors[fa], reoevalues[fa], &eigenCalculator));
      debug("reoevalues[%d] = %f", fa, reoevalues[fa]);
    }
    return std::move(evectorsUPtr);
  }

  static void launchDataTest(const oap::DataLoader::Info& info, const std::string& testFilename,
                             int wantedEigensCount = 5, int maxIterationCount = 1)
  {
    debugLongTest();
    try {
      std::string trace1;
      std::string trace2;

      initTraceBuffer(1024);
      MatricesUPtr deviceEVectors = TestCuHArnoldiCallback::launchTest(ArnUtils::DEVICE, info, wantedEigensCount, maxIterationCount);
      getTraceOutput(trace1);

      initTraceBuffer(1024);
      MatricesUPtr hostEVectors = TestCuHArnoldiCallback::launchTest(ArnUtils::HOST, info, wantedEigensCount, maxIterationCount);
      getTraceOutput(trace2);

      std::string pathTestDir = "/tmp/Oap/device_tests/";
      std::string pathTraceFiles = pathTestDir;
      std::string pathMatrixFiles = pathTestDir;
      pathTraceFiles += testFilename;
      pathMatrixFiles += "matrix";

      EXPECT_THAT(trace1, StringIsEqual(trace2, pathTraceFiles + "_DEVICE.log", pathTraceFiles + "_HOST.log"));

      math::Matrix** deviceMatrices = deviceEVectors.get();
      math::Matrix** hostMatrices = hostEVectors.get();
      math::Matrix* hostMatrix = oap::host::NewMatrix(hostEVectors.get()[0]);
      for (int fa = 0; fa < wantedEigensCount; ++fa) {
        oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, deviceMatrices[fa]);
        
        oap::host::PrintMatrixToFile(pathMatrixFiles + "_" + std::to_string(fa) + ".txt", hostMatrix);
        EXPECT_THAT(hostMatrices[fa], MatrixIsEqual(hostMatrix, InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));
      }
      oap::host::DeleteMatrix(hostMatrix);
    } catch (const std::exception& ex) {
      debugException(ex);
      throw;
    }
  }

 private:
  ArnoldiOperations* m_ao;
  const int m_counterLimit;
  int m_counter;
};

TEST_F(OapEigenCalculatorTests, NotInitializedTest) {
  TestCuHArnoldiCallback cuharnoldi(nullptr);
  EigenCalculator eigenCalc(&cuharnoldi);
  EXPECT_THROW(eigenCalc.calculate(), oap::exceptions::NotInitialzed);
}

TEST_F(OapEigenCalculatorTests, DISABLED_CalculateDeviceOutput) {
  oap::DataLoader::Info info("oap2dt3d/data/images_monkey", "image", 1000, true);
  TestCuHArnoldiCallback::launchDataTest(info, "CalculateDeviceOutput");
}

TEST_F(OapEigenCalculatorTests, DISABLED_CalculateDeviceOutput1) {
  oap::DataLoader::Info info("oap2dt3d/data/images_monkey_1", "image_", 64, true);
  TestCuHArnoldiCallback::launchDataTest(info, "CalculateDeviceOutput1", 10, 10);
}
