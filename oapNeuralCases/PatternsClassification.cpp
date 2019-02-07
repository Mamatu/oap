#include "PatternsClassification.h"

#include "CuProceduresApi.h"
#include "KernelExecutor.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "Controllers.h"

#include "Config.h"

#include <unistd.h>
#include <getopt.h>

namespace oap
{
namespace classification
{

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

int run_PatternsClassification (int argc, char **argv)
{

  return 0;
}

int run_PatternsClassification (const Args& args)
{
  auto load = [&args] (const std::string& path) -> std::unique_ptr<floatt[]>
  {
    oap::PngFile png (path, false);
    png.loadBitmap ();

    if (args.m_onOpenFile)
    {
      args.m_onOpenFile (png.getWidth (), png.getHeight (), png.isLoaded ());
    }
  
    std::unique_ptr<floatt[]> mask (new floatt[png.getLength()]);
    png.getFloattVector (mask.get ());

    return std::move (mask);
  };

  auto patternA = load (utils::Config::getFileInOap(args.patternPath1));
  auto patternB = load (utils::Config::getFileInOap(args.patternPath2));

  Network network;

  for (int layerSize : args.networkLayers)
  {
    network.createLayer (layerSize);
  }

  oap::HostMatrixPtr input = oap::host::NewReMatrix (1, args.networkLayers.front(), 0);
  oap::HostMatrixPtr eoutput = oap::host::NewReMatrix (1, args.networkLayers.back(), 0);

  SE_CD_Controller selc (0.001, 100);

  network.setLearningRate (0.001);
  network.setController (&selc);

  Network::ErrorType errorType = args.errorTpe;

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

  auto invokeCallback = [&args](const oap::HostMatrixUPtr& matrix, const Args::OutputCallback& callback)
  {
    std::vector<floatt> vec;
    for (size_t idx = 0; idx < args.networkLayers.back(); ++idx)
    {
      vec.push_back (matrix->reValues[idx]);
    }
    if (callback)
    {
      callback (vec);
    }
  };

  oap::host::CopyBuffer (input->reValues, patternA.get (), input->columns * input->rows);
  auto output1 = network.run (input, Network::HOST, errorType);
  invokeCallback (output1, args.m_onOutput1);

  oap::host::CopyBuffer (input->reValues, patternB.get (), input->columns * input->rows);
  auto output2 = network.run (input, Network::HOST, errorType);
  invokeCallback (output2, args.m_onOutput2);

  return 0;
}

int run_PatternsClassification ()
{
  return run_PatternsClassification (Args());
}
        
}
}
