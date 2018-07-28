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

#include <stdlib.h>
#include <string>

#include "Config.h"
#include "PngFile.h"
#include "oapCudaMatrixUtils.h"

#include "KernelExecutor.h"
#include "CuProceduresApi.h"

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixUPtr.h"

#include "oapNetwork.h"

#include <vector>
#include <cmath>
#include <utility>
#include <random>

Network* createNetwork (size_t width, size_t height)
{
  Network* network = new Network();

  network->createLayer(width*height);
  network->createLayer(15);
  network->createLayer(10);

  return network;
}

std::string getImagesPath()
{
  std::string dataPath = utils::Config::getPathInOap("oapNeural/data/");
  dataPath = dataPath + "digits/";
  return dataPath;
}

oap::HostMatrixUPtr getImageMatrix (const std::string& imagePath)
{
  oap::PngFile png(imagePath, false);
  png.open();
  png.loadBitmap();
  
  size_t width = png.getOutputWidth().optSize;
  size_t height = png.getOutputHeight().optSize;

  oap::HostMatrixUPtr input = oap::host::NewReMatrix (width, height);
  png.getFloattVector(input->reValues);

  return std::move (input);
};

Network* prepareNetwork(const std::vector<std::pair<std::string, int>>& dataSet)
{
  Network* network = nullptr;

  std::string dataPath = getImagesPath ();

  auto getImagePath = [&](size_t idx)
  {
    auto pair = dataSet[idx];
    std::string imagePath = dataPath + pair.first;
    return imagePath;
  };

  auto getImageMatrixFromIdx = [&](size_t idx)
  {
    auto path = getImagePath (idx);
    return getImageMatrix (path);
  };

  auto runTest = [&](math::Matrix* matrix, size_t idx)
  {
    math::Matrix* eoutput = oap::host::NewReMatrix (1, 10);
    auto pair = dataSet[idx];

    if (pair.second > -1)
    {
      eoutput->reValues[pair.second] = 1;
    }

    network->runHostArgsTest (matrix, eoutput);
    oap::host::DeleteMatrix (eoutput);
  };

  auto imatrix = getImageMatrixFromIdx (0);

  network = createNetwork (imatrix->columns, imatrix->rows);
  runTest (imatrix, 0);
/*
  for (size_t idx = 1; idx < dataSet.size(); ++idx)
  {
    oap::HostMatrixUPtr imatrix = getImageMatrixFromIdx (idx);

    runTest (imatrix, idx);
  }
*/
  return network;
}

int main()
{
  std::vector<std::pair<std::string, int>> dataSet = 
  {
   {"bias_1.png", -1},
   {"i1_1.png", 1},
   {"i2_1.png", 2},
   {"i2_2.png", 2},
   {"i3_1.png", 3},
   {"i3_2.png", 3},
   {"i4_1.png", 4},
   {"i4_2.png", 4},
   {"i5_1.png", 5},
   {"i6_1.png", 6},
   {"i7_1.png", 7},
   {"i8_1.png", 8},
   {"i8_2.png", 8},
   {"i8_3.png", 8},
   {"i9_1.png", 9},
   {"i9_2.png", 9},
   {"i9_3.png", 9},
   {"i9_4.png", 9},
   {"i9_5.png", 9},
   {"i9_6.png", 9}
  };

  oap::cuda::Context::Instance().create();

  Network* network = prepareNetwork (dataSet);
/*  std::string dataPath = getImagesPath ();

  oap::HostMatrixUPtr imatrix = getImageMatrix (dataPath + "i8_3.png");
  auto output = network->runHostArgs (imatrix.get());
  oap::host::PrintMatrix (output.get ());

  delete network;*/
  oap::cuda::Context::Instance().destroy();
  return 0;
}
