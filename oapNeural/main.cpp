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

#include <stdlib.h>
#include <string>

#include "Config.h"
#include "PngFile.h"
#include "oapCudaMatrixUtils.h"

#include "KernelExecutor.h"
#include "CuProceduresApi.h"

#include "oapHostMatrixPtr.h"
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
  network->createLayer(80);
  network->createLayer(10);

  return network;
}

std::string getImagesPath()
{
  std::string dataPath = utils::Config::getPathInOap("oapNeural/data/");
  dataPath = dataPath + "digits/";
  return dataPath;
}

oap::HostMatrixPtr getImageMatrix (const std::string& imagePath)
{
  oap::PngFile png(imagePath, false);
  png.open();
  png.loadBitmap();

  size_t width = png.getOutputWidth().optSize;
  size_t height = png.getOutputHeight().optSize;

  oap::HostMatrixUPtr imageMatrix = oap::host::NewReMatrix (width, height);
  //math::Matrix* imageMatrix = oap::host::NewReMatrix (width, height);
  png.getFloattVector(imageMatrix->reValues);

  oap::HostMatrixPtr input = oap::host::NewReMatrix (1, width * height);
  //math::Matrix* input = oap::host::NewReMatrix (1, width * height);

  oap::host::CopyReBuffer (input, imageMatrix);

  return input;
};

class Context final
{
  public:
    Network* network = nullptr;
    std::vector<oap::HostMatrixPtr> matrices;
    std::vector<std::pair<std::string, int>> dataSet;

    ~Context ()
    {
      delete network;
      matrices.clear ();
      dataSet.clear ();
    }
};

Context* init (const std::vector<std::pair<std::string, int>>& dataSet)
{
  Context* ctx = new Context ();;
  ctx->dataSet = dataSet;


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

  for (size_t idx = 0; idx < dataSet.size(); ++idx)
  {
    oap::HostMatrixPtr imatrix = getImageMatrixFromIdx (idx);

    if (ctx->network == nullptr)
    {
      ctx->network = createNetwork (imatrix->columns, imatrix->rows);
    }

    ctx->matrices.push_back (imatrix);
  }

  return ctx;
}

void runTraining (Context* ctx, floatt learningRate, int repeats)
{
  auto runTest = [&](math::Matrix* matrix, size_t idx)
  {
    math::Matrix* eoutput = oap::host::NewReMatrix (10, 1);
    auto pair = ctx->dataSet[idx];

    if (pair.second > -1)
    {
      eoutput->reValues[pair.second] = 1;
    }

    ctx->network->train (matrix, eoutput, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
    oap::host::DeleteMatrix (eoutput);
  };

  ctx->network->setLearningRate (learningRate);

  for (size_t testIdx = 0; testIdx < repeats; ++testIdx)
  {
    for (size_t idx = 0; idx < ctx->matrices.size(); ++idx)
    {
      debugInfo ("Training idx %lu", idx + testIdx * ctx->matrices.size());
      auto imatrix = ctx->matrices[idx];
      runTest (imatrix, idx);
    }
    ctx->network->getLayer(0)->printHostWeights (true);
  }
}

int main(int argc, char** argv)
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

  Context* ctx = init (dataSet);

  runTraining (ctx, 0.01f, 10);

  std::string dataPath = getImagesPath ();

  auto run = [&](const std::string& image)
  {
    oap::HostMatrixPtr imatrix = getImageMatrix (dataPath + image);
    auto output = ctx->network->run (imatrix.get(), ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
    oap::host::PrintMatrix ("output = ", output.get ());
  };

  run("i9_4");

  delete ctx;
  oap::cuda::Context::Instance().destroy();
  return 0;
}
