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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "PngFile.h"
#include "IEigenCalculator.h"
#include "DeviceImagesLoader.h"
#include "Exceptions.h"
#include "MatchersUtils.h"

#include "ArnoldiProceduresImpl.h"
#include "oapCudaMatrixUtils.h"
#include "CuProceduresApi.h"
#include "oapDeviceComplexMatrixUPtr.h"
#include "oapHostComplexMatrixUPtr.h"

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

    void setEigenvectorsOutput(math::ComplexMatrix** eigenvecs, ArnUtils::Type type)
    {
      oap::IEigenCalculator::setEigenvectorsOutput (eigenvecs, type);
    }

    oap::DeviceImagesLoader* getImagesLoader() const
    {
      return oap::IEigenCalculator::getImagesLoader ();
    }
};

class ArnoldiOperations {
 public:
  oap::DeviceImagesLoader* m_dataLoader;
  math::ComplexMatrix* value;
  oap::CuProceduresApi cuProceduresApi;

  ArnoldiOperations(oap::DeviceImagesLoader* dataLoader)
      : m_dataLoader(dataLoader) {
    value = oap::cuda::NewDeviceReMatrix(1, 1);
  }

  ~ArnoldiOperations() { oap::cuda::DeleteDeviceMatrix(value); }

  static void multiplyFunc(math::ComplexMatrix* m_w, math::ComplexMatrix* m_v,
                           oap::CuProceduresApi& cuProceduresApi,
                           void* userData, oap::VecMultiplicationType mt)
  {
    if (mt == oap::VecMultiplicationType::TYPE_WV) {
      ArnoldiOperations* ao = static_cast<ArnoldiOperations*>(userData);
      oap::DeviceImagesLoader* dataLoader = ao->m_dataLoader;

      math::MatrixInfo matrixInfo = dataLoader->getMatrixInfo();

      for (uintt index = 0; index < matrixInfo.columns (); ++index) {
        math::ComplexMatrix* vec = dataLoader->createDeviceRowVector(index);

        //oap::cuda::PrintMatrix("vec =", vec);

        cuProceduresApi.dotProduct(ao->value, vec, m_v);
        oap::cuda::SetMatrix(m_w, ao->value, 0, index);

        oap::cuda::DeleteDeviceMatrix(vec);
      }
    }
  }

  bool verifyOutput (math::ComplexMatrix* vector, floatt value, EigenCalculator* eigenCalc)
  {
    math::ComplexMatrix* matrix = m_dataLoader->createMatrix();

    const uintt partSize = gColumns (matrix);

    math::ComplexMatrix* dvector = NULL;
    uintt vectorrows = 0;

    bool dvectorIsCopy = false;

    vectorrows = gColumns (matrix);

    if (eigenCalc->getEigenvectorsType() == ArnUtils::HOST) {
      dvector = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(vector);
      dvectorIsCopy = true;
    } else if (eigenCalc->getEigenvectorsType() == ArnUtils::DEVICE) {
      dvector = vector;
      dvectorIsCopy = false;
    }

    if (dvector == NULL)  {
      debugAssert("Invalid eigenvectors type.");
    }

    oap::HostComplexMatrixUPtr refMatrix = oap::host::NewComplexMatrix(matrix, gColumns (matrix), partSize);

    oap::host::CopyMatrix(refMatrix, matrix);

    oap::DeviceComplexMatrixUPtr drefMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(refMatrix);

    math::MatrixInfo info = oap::host::GetMatrixInfo(refMatrix);

    const uintt matrixcolumns = info.columns ();
    uintt matrixrows = info.rows ();

    matrixrows = partSize;

    oap::DeviceComplexMatrixUPtr matrix1 = oap::cuda::NewDeviceReMatrix(matrixrows, matrixcolumns);

    oap::DeviceComplexMatrixUPtr leftMatrix = oap::cuda::NewDeviceReMatrix(matrixrows, matrixrows);
    oap::DeviceComplexMatrixUPtr rightMatrix = oap::cuda::NewDeviceReMatrix(matrixrows, matrixrows);

    oap::DeviceComplexMatrixUPtr vectorT = oap::cuda::NewDeviceReMatrix(vectorrows, 1);

    cuProceduresApi.transpose(matrix1, drefMatrix);
    cuProceduresApi.transpose(vectorT, dvector);
    cuProceduresApi.dotProduct(leftMatrix, drefMatrix, matrix1);

    floatt value2 = value * value;
    cuProceduresApi.multiplyReConstant(vectorT, vectorT, value2);
    cuProceduresApi.dotProduct(rightMatrix, dvector, vectorT);
    bool compareResult = cuProceduresApi.compare(leftMatrix, rightMatrix);

    oap::HostComplexMatrixUPtr hleftMatrix = oap::host::NewReMatrix(oap::cuda::GetColumns(leftMatrix), oap::cuda::GetRows(leftMatrix));
    oap::HostComplexMatrixUPtr hrightMatrix = oap::host::NewReMatrix(oap::cuda::GetColumns(rightMatrix), oap::cuda::GetRows(rightMatrix));

    oap::cuda::CopyDeviceMatrixToHostMatrix(hrightMatrix, rightMatrix);
    oap::cuda::CopyDeviceMatrixToHostMatrix(hleftMatrix, leftMatrix);

    EXPECT_THAT(hleftMatrix.get(), MatrixIsEqual(hrightMatrix.get(), InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));

    oap::host::DeleteMatrix(matrix);
    if (dvectorIsCopy) {
      oap::cuda::DeleteDeviceMatrix(dvector);
    }

    return compareResult;
  }

  math::ComplexMatrix* createDeviceMatrix() const {
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

    MatricesDeleter& operator() (math::ComplexMatrix** evectors) {
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

using MatricesUPtr = std::unique_ptr<math::ComplexMatrix*, MatricesDeleter>;

class TestCuHArnoldiCallback : public CuHArnoldiCallback {
 public:
  TestCuHArnoldiCallback(ArnoldiOperations* ao, int counterLimit = 5) : m_ao(ao), m_counterLimit(counterLimit), m_counter(0) {
  }

  bool checkEigenspair(floatt revalue, floatt imvalue, math::ComplexMatrix* vector, uint index, uint max) {
    ++m_counter;
    debug("counter = %d", m_counter);

    bool output = (m_counter < m_counterLimit);
    if (output == false && index == max - 1) {
      m_counter = 0;
    }
    return output;
  }

  static MatricesUPtr launchTest(ArnUtils::Type eigensType, const oap::ImagesLoader::Info& info,
                                 int wantedEigensCount, int maxIterationCounter = 5)
  {
    std::unique_ptr<oap::DeviceImagesLoader> dataLoader(
        oap::DeviceImagesLoader::createImagesLoader<oap::PngFile, oap::DeviceImagesLoader>(info));

    ArnoldiOperations ao(dataLoader.get());
    TestCuHArnoldiCallback cuharnoldi(&ao, maxIterationCounter);
    cuharnoldi.setCallback(ArnoldiOperations::multiplyFunc, &ao);

    floatt reoevalues[wantedEigensCount];

    EigenCalculator eigenCalculator(&cuharnoldi);

    eigenCalculator.setImagesLoader (dataLoader.get());

    eigenCalculator.setEigensCount(wantedEigensCount, wantedEigensCount);

    math::MatrixInfo matrixInfo = eigenCalculator.getMatrixInfo();

    MatricesDeleter matricesDeleter(wantedEigensCount, eigensType);

    MatricesUPtr evectorsUPtr(new math::ComplexMatrix* [wantedEigensCount], matricesDeleter);

    auto matricesInitializer = [&evectorsUPtr, wantedEigensCount, &matrixInfo, eigensType]() {
      math::ComplexMatrix** evectors = evectorsUPtr.get();
      const uintt rows = matrixInfo.rows ();
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

    math::ComplexMatrix** evectors = evectorsUPtr.get();

    eigenCalculator.setEigenvaluesOutput(reoevalues);

    eigenCalculator.setEigenvectorsOutput(evectors, eigensType);

    eigenCalculator.calculate();

    for (int fa = 0; fa < wantedEigensCount; ++fa) {
      EXPECT_TRUE(ao.verifyOutput(evectors[fa], reoevalues[fa], &eigenCalculator));
      debug("reoevalues[%d] = %f", fa, reoevalues[fa]);
    }
    return std::move(evectorsUPtr);
  }

  static void launchDataTest(const oap::ImagesLoader::Info& info, const std::string& testFilename,
                             int wantedEigensCount = 5, int maxIterationCount = 1)
  {
    logInfoLongTest();
    try {
      std::string trace1;
      std::string trace2;

      MatricesUPtr deviceEVectors = TestCuHArnoldiCallback::launchTest(ArnUtils::DEVICE, info, wantedEigensCount, maxIterationCount);
      MatricesUPtr hostEVectors = TestCuHArnoldiCallback::launchTest(ArnUtils::HOST, info, wantedEigensCount, maxIterationCount);

      std::string pathTestDir = "/tmp/Oap/device_tests/";
      std::string pathTraceFiles = pathTestDir;
      std::string pathMatrixFiles = pathTestDir;
      pathTraceFiles += testFilename;
      pathMatrixFiles += "matrix";

      EXPECT_THAT(trace1, StringIsEqual(trace2, pathTraceFiles + "_DEVICE.log", pathTraceFiles + "_HOST.log"));

      math::ComplexMatrix** deviceMatrices = deviceEVectors.get();
      math::ComplexMatrix** hostMatrices = hostEVectors.get();
      math::ComplexMatrix* hostMatrix = oap::host::NewComplexMatrixRef (hostEVectors.get()[0]);
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
  oap::ImagesLoader::Info info("oap2dt3d/data/images_monkey", "image", 1000, true);
  TestCuHArnoldiCallback::launchDataTest(info, "CalculateDeviceOutput");
}

TEST_F(OapEigenCalculatorTests, DISABLED_CalculateDeviceOutput1) {
  oap::ImagesLoader::Info info("oap2dt3d/data/images_monkey_1", "image_", 64, true);
  TestCuHArnoldiCallback::launchDataTest(info, "CalculateDeviceOutput1", 10, 10);
}
