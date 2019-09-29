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

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "Config.h"

#include "oapHostMatrixUtils.h"
#include "HostProcedures.h"

#include "oapGenericArnoldiApi.h"
#include "oapCuHArnoldiS.h"

class OapArnoldiPackageCallbackTests : public testing::Test {
  public:

    virtual void SetUp() {}
    virtual void TearDown() {}

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
        std::string absdir = utils::Config::getPathInOap("oapArnoldiDeviceTests") + dir;
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

        refV = oap::host::NewMatrix(true, true, 1, m_elementsCount);
        refW = oap::host::NewMatrix(true, true, 1, m_elementsCount);
        hostV = oap::host::NewMatrix(true, true, 1, m_elementsCount);
        hostW = oap::host::NewMatrix(true, true, 1, m_elementsCount);
      }

      virtual ~Data() {
        oap::host::DeleteMatrix(refV);
        oap::host::DeleteMatrix(refW);
        oap::host::DeleteMatrix(hostV);
        oap::host::DeleteMatrix(hostW);
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

    void executeArnoldiTest (floatt value, const std::string& path,
                             uintt hdim = 32, bool enableValidateOfV = true,
                             floatt tolerance = 0.01)
    {
      OapArnoldiPackageCallbackTests::Data data(path);

      using UserPair = std::pair<OapArnoldiPackageCallbackTests::Data*, bool>;

      UserPair userPair = std::make_pair(&data, enableValidateOfV);

      auto multiply = [&userPair] (math::Matrix* w, math::Matrix* v, HostProcedures& cuProceduresApi, oap::VecMultiplicationType mt)
      {
        if (mt == oap::VecMultiplicationType::TYPE_WV) {
          Data* data = userPair.first;
          data->load();
          oap::host::CopyHostMatrixToHostMatrix(data->hostV, v);
          if (userPair.second) {
            ASSERT_THAT(data->hostV,
                        MatrixIsEqual(
                            data->refV,
                            InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));
          }
          oap::host::CopyHostMatrixToHostMatrix(w, data->refW);
        }
      };

      oap::generic::CuHArnoldiS ca;
      HostProcedures hp;
      math::MatrixInfo matrixInfo (true, true, data.getElementsCount(), data.getElementsCount());

      oap::generic::allocStage1 (ca, matrixInfo, oap::host::NewHostMatrix);
      oap::generic::allocStage2 (ca, matrixInfo, 32, oap::host::NewHostMatrix, oap::host::NewHostMatrix);
      oap::generic::allocStage3 (ca, matrixInfo, 32, oap::host::NewHostMatrix, oap::QRType::QRGR);

      oap::generic::iram_executeInit (ca, hp, multiply);

      oap::generic::deallocStage1 (ca, oap::host::DeleteMatrix);
      oap::generic::deallocStage2 (ca, oap::host::DeleteMatrix, oap::host::DeleteMatrix);
      oap::generic::deallocStage3 (ca, oap::host::DeleteMatrix);
    }
};

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
