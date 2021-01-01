#include "PatternsClassification.h"

#include "CuProceduresApi.h"
#include "KernelExecutor.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "Controllers.h"

#include "Config.h"
#include "MatrixAPI.h"

#include "ArgsParser.h"

namespace oap
{

template<typename Callback, typename CallbackNL>
void iterateBitmap (floatt* pixels, const oap::ImageSection& width, const oap::ImageSection& height, Callback&& callback, CallbackNL&& cnl)
{
  for (size_t y = 0; y < height.getl(); ++y)
  {
    for (size_t x = 0; x < width.getl(); ++x)
    {
      floatt value = pixels[x + width.getl() * y];
      int pvalue = value > 0.5 ? 1 : 0;
      callback (pvalue, x, y);
    }
    cnl ();
  }
  cnl ();
}

void printBitmap (floatt* pixels, const oap::ImageSection& width, const oap::ImageSection& height)
{
  iterateBitmap (pixels, width, height, [](int pixel, size_t x, size_t y){ printf ("%d", pixel); }, [](){ printf("\n"); });
}

PatternsClassification::PatternsClassification () : m_parser (), m_bInterrupted (false)
{}

PatternsClassification::~PatternsClassification ()
{}

int PatternsClassification::run ()
{
  return PatternsClassification::run (PatternsClassificationParser::Args ());
}

int PatternsClassification::run (const oap::PatternsClassificationParser::Args& args)
{
  if (args.networkLayers.empty ())
  {
    return 1;
  }

  auto load = [&args] (const std::string& path) -> std::tuple<std::unique_ptr<floatt[]>, oap::ImageSection, oap::ImageSection>
  {
    oap::PngFile png (path, false);
    png.loadBitmap ();

    if (args.m_onOpenFile)
    {
      args.m_onOpenFile (png.getWidth (), png.getHeight (), png.isLoaded ());
    }
  
    std::unique_ptr<floatt[]> bitmap (new floatt[png.getLength()]);
    png.getFloattVector (bitmap.get ());

    auto width = png.getOutputWidth ();
    auto height = png.getOutputHeight ();

    return std::make_tuple (std::move (bitmap), width, height);
  };

  auto patternA = load (oap::utils::Config::getFileInOap(args.patternPath1));
  auto patternB = load (oap::utils::Config::getFileInOap(args.patternPath2));

  oap::cuda::Context::Instance().create();

  Network* network = nullptr;

  if (args.loadingPath.empty ())
  {
    oapAssert ("to check" == nullptr);
    //network = new Network();
  }
  else
  {
    utils::ByteBuffer buffer;
    buffer.fread (args.loadingPath);
    //network = Network::load (buffer);
  }

  for (int layerSize : args.networkLayers)
  {
    network->createLayer (layerSize);
  }

  oap::HostMatrixPtr input = oap::host::NewReMatrixWithValue (1, args.networkLayers.front(), 0);
  oap::HostMatrixPtr eoutput = oap::host::NewReMatrixWithValue (1, args.networkLayers.back(), 0);

  network->setLearningRate (0.001);

  oap::ErrorType errorType = args.errorType;

  auto upatternA = std::move (std::get<0>(patternA));
  auto upatternB = std::move (std::get<0>(patternB));

  printBitmap (upatternA.get(), std::get<1>(patternA).getl(), std::get<2>(patternA).getl());
  printBitmap (upatternB.get(), std::get<1>(patternB).getl(), std::get<2>(patternB).getl());

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis(0., 1.);

  while (!m_bInterrupted)
  {
    if (dis(dre) >= 0.5)
    {
      oap::host::CopyBuffer (input->re.ptr, upatternA.get (), gColumns (input) * gRows (input));
      *GetRePtrIndex (eoutput, 0) = 1;
    }
    else
    {
      oap::host::CopyBuffer (input->re.ptr, upatternB.get (), gColumns (input) * gRows (input));
      *GetRePtrIndex (eoutput, 0) = 0;
    }

    network->setExpected (eoutput, ArgType::HOST);
    network->setInputs (input, ArgType::HOST);
    network->forwardPropagation ();
    network->accumulateErrors (errorType, CalculationType::HOST);
    network->backPropagation();
    floatt error = network->calculateError (errorType);
    network->updateWeights ();
    logInfo ("error = %f", error);
  }

  auto invokeCallback = [&args](const oap::HostMatrixUPtr& matrix, const oap::PatternsClassificationParser::Args::OutputCallback& callback)
  {
    std::vector<floatt> vec;
    for (size_t idx = 0; idx < args.networkLayers.size(); ++idx)
    {
      vec.push_back (GetReIndex (matrix.get(), idx));
    }
    if (callback)
    {
      callback (vec);
    }
  };

  if (!m_bInterrupted)
  {
    oap::host::CopyBuffer (input->re.ptr, upatternA.get (), gColumns (input) * gRows (input));
    auto output1 = network->run (input, ArgType::HOST, errorType);
    invokeCallback (output1, args.m_onOutput1);

    oap::host::CopyBuffer (input->re.ptr, upatternB.get (), gColumns (input) * gRows (input));
    auto output2 = network->run (input, ArgType::HOST, errorType);
    invokeCallback (output2, args.m_onOutput2);
  }
  else
  {
    if (!args.savingPath.empty ())
    {
      utils::ByteBuffer buffer;
      //network->save (buffer);
      buffer.fwrite (args.savingPath);
    }
  }

  delete network;
  oap::cuda::Context::Instance().destroy();

  m_cond.signal ();

  return 0;
}

int PatternsClassification::runRoutine ()
{
  return PatternsClassification::run (m_parser.getArgs ());
}

const oap::IArgsParser& PatternsClassification::getArgsParser() const
{
  return m_parser;
}

void PatternsClassification::onInterrupt()
{
  debugFunc ();
  m_bInterrupted = true;
  debugFunc ();
  m_cond.wait ();
  debugFunc ();
}

void fclose_safe (FILE* file)
{
  if (file != nullptr)
  {
    debug ("Fclose of %p", file);
    fclose (file);
  }
}

void checkIfFileExists (FILE* file, const std::string& path)
{
  if (!file)
  {
    std::stringstream sstream;
    sstream << "File \"" << path << "\" cannot be open to read.";
    throw std::runtime_error (sstream.str ());
  }
}

}
