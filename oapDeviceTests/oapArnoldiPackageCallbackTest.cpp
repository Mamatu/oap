/*
 * Copyright 2016, 2017 Marcin Matula
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

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "Config.h"
#include "KernelExecutor.h"
#include "HostMatrixUtils.h"
#include "DeviceMatrixModules.h"
#include "MathOperationsCpu.h"
#include "matrix1.h"
#include "matrix2.h"
#include "matrix3.h"
#include "matrix4.h"
#include "matrix5.h"

#include "ArnoldiProceduresImpl.h"

class OapArnoldiPackageCallbackTests : public testing::Test {
  public:
    CuHArnoldiCallback* arnoldiCuda;

    virtual void SetUp() {
      device::Context::Instance().create();

      arnoldiCuda = new CuHArnoldiCallback();
      arnoldiCuda->setOutputType(ArnUtils::HOST);
    }

    virtual void TearDown() {
      delete arnoldiCuda;
      device::Context::Instance().destroy();
    }

    class Data {
      int m_counter;
      std::string m_vredir;
      std::string m_vimdir;
      std::string m_wredir;
      std::string m_wimdir;

      int m_blocksCount;
      int m_elementsCount;
      int m_size;

     public:
      Data(const std::string& dir)
          : m_counter(0), refV(NULL), hostV(NULL), refW(NULL), hostW(NULL) {
        std::string absdir = utils::Config::getPathInOap("oapDeviceTests") + dir;
        m_vredir = absdir + "/vre.tdata";
        m_vimdir = absdir + "/vim.tdata";
        m_wredir = absdir + "/wre.tdata";
        m_wimdir = absdir + "/wim.tdata";

        FILE* file = fopen(m_vredir.c_str(), "rb");

        if (file == NULL) {
          std::stringstream ss;
          ss << "File " << m_vredir << " does not exist.";
          debugAssert(ss.str().c_str());
        }

        m_blocksCount = loadBlocksCount(file);
        m_elementsCount = loadElementsCount(file);
        m_size = loadSize(file);

        fclose(file);

        refV = host::NewMatrix(true, true, 1, m_elementsCount);
        refW = host::NewMatrix(true, true, 1, m_elementsCount);
        hostV = host::NewMatrix(true, true, 1, m_elementsCount);
        hostW = host::NewMatrix(true, true, 1, m_elementsCount);
      }

      virtual ~Data() {
        host::DeleteMatrix(refV);
        host::DeleteMatrix(refW);
        host::DeleteMatrix(hostV);
        host::DeleteMatrix(hostW);
      }

      void load() {
        loadBlock(m_vredir, refV->reValues, m_counter);
        loadBlock(m_vimdir, refV->imValues, m_counter);
        loadBlock(m_wredir, refW->reValues, m_counter);
        loadBlock(m_wimdir, refW->imValues, m_counter);
        ++m_counter;
      }

      int getElementsCount() const { return m_elementsCount; }

      int getCounter() const { return m_counter; }

      void printCounter() const { printf("Counter = %d \n", m_counter); }

      math::Matrix* refV;
      math::Matrix* hostV;
      math::Matrix* refW;
      math::Matrix* hostW;
    };

    static int loadBlocksCount(FILE* f) {
      int counter = 0;
      fseek(f, 2 * sizeof(int), SEEK_SET);
      fread(&counter, sizeof(int), 1, f);
      return counter;
    }

    static int loadSize(FILE* f) {
      if (f == NULL) {
        return -1;
      }
      int size = 0;
      fseek(f, 0, SEEK_SET);
      fread(&size, sizeof(int), 1, f);
      return size;
    }

    static int loadElementsCount(FILE* f) {
      if (f == NULL) {
        return -1;
      }
      int count = 0;
      fseek(f, sizeof(int), SEEK_SET);
      fread(&count, sizeof(int), 1, f);
      return count;
    }

    template <typename T>
    static void copySafely(floatt* block, int size, int elementsCount, FILE* f) {
      T* tmpBuffer = new T[elementsCount];
      fread(tmpBuffer, elementsCount * size, 1, f);
      for (uintt fa = 0; fa < elementsCount; ++fa) {
        block[fa] = tmpBuffer[fa];
      }
      delete[] tmpBuffer;
    }

    static void readBlock(floatt* block, int size, int elementsCount, FILE* f) {
      if (sizeof(floatt) == size) {
        fread(block, elementsCount * size, 1, f);
      } else {
        if (size == 4) {
          copySafely<float>(block, size, elementsCount, f);
        } else if (size == 8) {
          copySafely<double>(block, size, elementsCount, f);
        } else {
          debugAssert("Size not implemented.");
        }
      }
    }

    static void loadBlock(FILE* f, floatt* block, int index) {
      int blocksCount = loadBlocksCount(f);
      int elementsCount = loadElementsCount(f);
      int size = loadSize(f);
      fseek(f, 3 * sizeof(int), SEEK_SET);
      fseek(f, index * elementsCount * size, SEEK_CUR);
      readBlock(block, size, elementsCount, f);
    }

    static void loadBlock(const std::string& path, floatt* block, int index) {
      FILE* f = fopen(path.c_str(), "rb");
      if (f == NULL) {
        debug("File does not exist: %s", path.c_str());
        abort();
      }
      loadBlock(f, block, index);
      fclose(f);
    }

    void executeArnoldiTest(floatt value, const std::string& path,
                            uintt hdim = 32, bool enableValidateOfV = true,
                            floatt tolerance = 0.01) {
      OapArnoldiPackageCallbackTests::Data data(path);

      typedef std::pair<OapArnoldiPackageCallbackTests::Data*, bool> UserPair;

      UserPair userPair = std::make_pair(&data, enableValidateOfV);

      class MultiplyFunc {
       public:
        static void multiply(math::Matrix* w, math::Matrix* v,
                             CuMatrix& cuProceduresApi,
                             void* userData,
                             CuHArnoldi::MultiplicationType mt) {
          if (mt == CuHArnoldi::TYPE_WV) {
            UserPair* userPair = static_cast<UserPair*>(userData);
            Data* data = userPair->first;
            data->load();
            device::CopyDeviceMatrixToHostMatrix(data->hostV, v);
            if (userPair->second) {
              ASSERT_THAT(data->hostV,
                          MatrixIsEqual(
                              data->refV,
                              InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));
            }
            device::CopyHostMatrixToDeviceMatrix(w, data->refW);
          }
        }
      };
      floatt revalues[2] = {0, 0};
      floatt imvalues[2] = {0, 0};

      uintt wanted = 1;

      arnoldiCuda->setCallback(MultiplyFunc::multiply, &userPair);
      arnoldiCuda->setBLimit(0.01);
      arnoldiCuda->setRho(1. / 3.14159265359);
      arnoldiCuda->setSortType(ArnUtils::SortSmallestReValues);
      arnoldiCuda->setCheckType(ArnUtils::CHECK_FIRST_STOP);
      arnoldiCuda->setOutputsEigenvalues(revalues, imvalues);
      math::MatrixInfo matrixInfo(true, true, data.getElementsCount(),
                                      data.getElementsCount());

      debugLongTest();

      arnoldiCuda->execute(hdim, wanted, matrixInfo);
      EXPECT_THAT(revalues[0], ::testing::DoubleNear(value, tolerance));
      // EXPECT_DOUBLE_EQ(value, );
      EXPECT_DOUBLE_EQ(revalues[1], 0);
      EXPECT_DOUBLE_EQ(imvalues[0], 0);
      EXPECT_DOUBLE_EQ(imvalues[1], 0);
    }

    void triangularityTest(const std::string& matrixStr) {
      math::Matrix* matrix = host::NewMatrix(matrixStr);
      triangularityTest(matrix);
      host::DeleteMatrix(matrix);
    }

    void triangularityTest(const math::Matrix* matrix) {
      floatt limit = 0.001;
      for (int fa = 0; fa < matrix->columns - 1; ++fa) {
        floatt value = matrix->reValues[(fa + 1) * matrix->columns + fa];
        bool islower = value < limit;
        bool isgreater = -limit < value;
        EXPECT_TRUE(islower) << value << " is greater than " << limit
                             << " index= " << fa << ", " << fa + 1;
        EXPECT_TRUE(isgreater) << value << " is lower than " << -limit
                               << " index= " << fa << ", " << fa + 1;
      }
    }
};

TEST_F(OapArnoldiPackageCallbackTests, MagnitudeTest) {
  OapArnoldiPackageCallbackTests::Data data("data/data1");
  data.load();

  bool isre = data.refW->reValues != NULL;
  bool isim = data.refW->imValues != NULL;

  uintt columns = data.refW->columns;
  uintt rows = data.refW->rows;

  math::Matrix* dmatrix = device::NewDeviceMatrix(isre, isim, columns, rows);

  device::CopyHostMatrixToDeviceMatrix(dmatrix, data.refW);

  CuMatrix cuProceduresApi;

  floatt output = -1;
  floatt doutput = -1;
  cuProceduresApi.magnitude(doutput, dmatrix);

  math::MathOperationsCpu mocpu;
  mocpu.magnitude(&output, data.refW);

  EXPECT_DOUBLE_EQ(3.25, output);
  EXPECT_DOUBLE_EQ(3.25, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);

  device::DeleteDeviceMatrix(dmatrix);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData1) {
  executeArnoldiTest(-3.25, "data/data1");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData2Dim32x32) {
  executeArnoldiTest(-4.257104, "data/data2", 32);
}

TEST_F(OapArnoldiPackageCallbackTests, DISABLED_TestData2Dim64x64) {
  executeArnoldiTest(-4.257104, "data/data2", 64, false);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData3Dim32x32) {
  executeArnoldiTest(-5.519614, "data/data3", 32);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData3Dim64x64) {
  executeArnoldiTest(-5.519614, "data/data3", 64, false);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData4) {
  executeArnoldiTest(-6.976581, "data/data4");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData5) {
  executeArnoldiTest(-8.503910, "data/data5");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData6) {
  executeArnoldiTest(-10.064733, "data/data6");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData7) {
  executeArnoldiTest(-13.235305, "data/data7");
}
