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

#ifndef OAP_TEST_DATA_LOADER_H
#define OAP_TEST_DATA_LOADER_H

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "MatchersUtils.h"
#include "Config.h"

#include "oapHostMatrixUtils.h"
#include "HostProcedures.h"

#include "oapGenericArnoldiApi.h"
#include "oapCuHArnoldiS.h"
#include "oapAssertion.h"

namespace oap
{
class ACTestData
{
  int m_counter;
  std::string m_vredir;
  std::string m_vimdir;
  std::string m_wredir;
  std::string m_wimdir;

  int m_blocksCount;
  int m_elementsCount;
  int m_size;

 public:
  static int loadBlocksCount(FILE* f)
  {
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");
  
    int counter = 0;
    fseek(f, 2 * sizeof(int), SEEK_SET);
    fread(&counter, sizeof(int), 1, f);
    return counter;
  }
  
  static int loadSize(FILE* f)
  {
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    int size = 0;
    fseek(f, 0, SEEK_SET);
    fread(&size, sizeof(int), 1, f);
    return size;
  }
  
  static int loadElementsCount(FILE* f)
  {
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    int count = 0;
    fseek(f, sizeof(int), SEEK_SET);
    fread(&count, sizeof(int), 1, f);
    return count;
  }
  
  template <typename T>
  static void copySafely(floatt* block, int size, int elementsCount, FILE* f)
  {
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    T* tmpBuffer = new T[elementsCount];
    fread(tmpBuffer, elementsCount * size, 1, f);
    for (uintt fa = 0; fa < elementsCount; ++fa) {
      block[fa] = tmpBuffer[fa];
    }
    delete[] tmpBuffer;
  }
  
  static void readBlock(floatt* block, int size, int elementsCount, FILE* f)
  {
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    if (sizeof(floatt) == size)
    {
      fread(block, elementsCount * size, 1, f);
    }
    else
    {
      if (size == 4) {
        copySafely<float>(block, size, elementsCount, f);
      } else if (size == 8) {
        copySafely<double>(block, size, elementsCount, f);
      } else {
        debugAssert("Size not implemented.");
      }
    }
  }
  
  static void loadBlock(FILE* f, floatt* block, int index)
  {
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    int blocksCount = loadBlocksCount(f);
    int elementsCount = loadElementsCount(f);
    int size = loadSize(f);
    fseek(f, 3 * sizeof(int), SEEK_SET);
    fseek(f, index * elementsCount * size, SEEK_CUR);
    readBlock(block, size, elementsCount, f);
  }
  
  static void loadBlock(const std::string& path, floatt* block, int index)
  {
    FILE* f = fopen(path.c_str(), "rb");
    logAssertMsg (f != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    loadBlock(f, block, index);
    fclose(f);
  }
    
  ACTestData(const std::string& dir)
      : m_counter(0), refV(NULL), hostV(NULL), refW(NULL), hostW(NULL) {
    std::string absdir = utils::Config::getPathInOap("oapDeviceTests") + dir;
    m_vredir = absdir + "/vre.tdata";
    m_vimdir = absdir + "/vim.tdata";
    m_wredir = absdir + "/wre.tdata";
    m_wimdir = absdir + "/wim.tdata";

    FILE* file = fopen(m_vredir.c_str(), "rb");
    logAssertMsg (file != nullptr, "%s", "Files witd data do not exist. Please check if data are available in oapDeviceTests/data");

    m_blocksCount = loadBlocksCount(file);
    m_elementsCount = loadElementsCount(file);
    m_size = loadSize(file);

    fclose(file);

    refV = oap::host::NewMatrix(true, true, 1, m_elementsCount);
    refW = oap::host::NewMatrix(true, true, 1, m_elementsCount);
    hostV = oap::host::NewMatrix(true, true, 1, m_elementsCount);
    hostW = oap::host::NewMatrix(true, true, 1, m_elementsCount);
  }

  virtual ~ACTestData() {
    oap::host::DeleteMatrix(refV);
    oap::host::DeleteMatrix(refW);
    oap::host::DeleteMatrix(hostV);
    oap::host::DeleteMatrix(hostW);
  }

  void load() {
    loadBlock(m_vredir, refV->re.ptr, m_counter);
    loadBlock(m_vimdir, refV->im.ptr, m_counter);
    loadBlock(m_wredir, refW->re.ptr, m_counter);
    loadBlock(m_wimdir, refW->im.ptr, m_counter);
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
}

#endif
