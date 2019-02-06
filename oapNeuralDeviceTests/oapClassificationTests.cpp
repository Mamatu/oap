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

#include <string>
#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "KernelExecutor.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "oapNetwork.h"
#include "Controllers.h"

#include "PngFile.h"
#include "Config.h"

class OapClassificationTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown()
  {
    oap::cuda::Context::Instance().destroy();
  }
};

template<typename Callback, typename CallbackNL>
void iterateBitmap (floatt* pixels, const oap::OptSize& width, const oap::OptSize& height, Callback&& callback, CallbackNL&& cnl)
{
  for (size_t y = 0; y < height.optSize; ++y)
  {
    for (size_t x = 0; x < width.optSize; ++x)
    {
      floatt value = pixels[x + width.optSize * y];
      int pvalue = value > 0.5 ? 1 : 0;
      callback (pvalue, x, y);
    }
    cnl ();
  }
  cnl ();
}

void printBitmap (floatt* pixels, const oap::OptSize& width, const oap::OptSize& height)
{
  iterateBitmap (pixels, width, height, [](int pixel, size_t x, size_t y){ printf ("%d", pixel); }, [](){ printf("\n"); });
}

TEST_F(OapClassificationTests, SimpleClassification)
{
  auto load = [] (const std::string& path) -> std::unique_ptr<floatt[]>
  {
    oap::PngFile png (path, false);
    png.loadBitmap ();

    EXPECT_EQ(20, png.getWidth().optSize);
    EXPECT_EQ(20, png.getHeight().optSize);
    EXPECT_TRUE(png.isLoaded ());
  
    std::unique_ptr<floatt[]> mask (new floatt[png.getLength()]);
    png.getFloattVector (mask.get ());

    return std::move (mask);
  };

  auto patternA = load (utils::Config::getFileInOap("oapNeural/data/text/a.png"));
  auto patternB = load (utils::Config::getFileInOap("oapNeural/data/text/b.png"));

  Network network;
  network.createLayer (20 * 20);
  network.createLayer (20);
  network.createLayer (1);

  oap::HostMatrixPtr input = oap::host::NewReMatrix (1, 20*20, 0);
  oap::HostMatrixPtr eoutput = oap::host::NewReMatrix (1, 1, 0);

  SE_CD_Controller selc (0.001, 100);

  network.setLearningRate (0.001);
  network.setController (&selc);

  Network::ErrorType errorType = Network::ErrorType::CROSS_ENTROPY;
  //Network::ErrorType errorType = Network::ErrorType::MEAN_SQUARE_ERROR;
  printBitmap (patternA.get(), 20, 20);
  printBitmap (patternB.get(), 20, 20);

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis(0., 1.);

  while (selc.shouldContinue())
  {
    if (dis(dre) >= 0.5)
    {
      oap::host::CopyBuffer (input->reValues, patternA.get (), input->columns * input->rows);
      eoutput->reValues[0] = 1;
    }
    else
    {
      oap::host::CopyBuffer (input->reValues, patternB.get (), input->columns * input->rows);
      eoutput->reValues[0] = 0;
    }

    network.train (input, eoutput, Network::HOST, errorType);
  }

  oap::host::CopyBuffer (input->reValues, patternA.get (), input->columns * input->rows);
  auto output = network.run (input, Network::HOST, errorType);
  EXPECT_LE(0.5, output->reValues[0]);

  oap::host::CopyBuffer (input->reValues, patternB.get (), input->columns * input->rows);
  auto output1 = network.run (input, Network::HOST, errorType);
  EXPECT_GE(0.5, output1->reValues[0]);
}
