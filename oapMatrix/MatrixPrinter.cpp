#include "MatrixPrinter.h"

namespace matrixUtils
{

void PrintReValues(std::string& output, const MatrixRange& matrixRange, const PrintArgs& args)
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;
  const std::string sectionSeparator = args.sectionSeparator;
  const size_t sectionLength = args.sectionLength;

  SubArrays<floatt> subArrrays;
  matrixRange.getReSubArrays(subArrrays);
  PrintArrays(output, subArrrays, args);
}

void PrintImValues(std::string& output, const MatrixRange& matrixRange, const PrintArgs& args)
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;
  const std::string sectionSeparator = args.sectionSeparator;
  const size_t sectionLength = args.sectionLength;

  SubArrays<floatt> subArrrays;
  matrixRange.getImSubArrays(subArrrays);
  PrintArrays(output, subArrrays, args);
}

void PrintMatrix(std::string& output, const MatrixRange& matrixRange, const PrintArgs& args)
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;
  const std::string sectionSeparator = args.sectionSeparator;

  std::stringstream sstream;
  const std::string reImSeparator[] = {" + ", "i * "};

  sstream << args.pretext;

  if (matrixRange.isReValues())
  {
    PrintArgs args1 = args;
    args1.pretext = "";
    args1.posttext = "";

    std::string loutput;
    PrintReValues(loutput, matrixRange, args1);

    sstream << loutput;
  }

  if (matrixRange.isImValues())
  {
    if (matrixRange.isReValues ())
    {
      sstream << reImSeparator[0];
    }
    sstream << reImSeparator[1];

    PrintArgs args1 = args;
    args1.pretext = "";
    args1.posttext = "";

    std::string loutput;
    PrintImValues(loutput, matrixRange, args1);

    sstream << loutput;
  }

  sstream << args.posttext;
  output = sstream.str();
}

void PrintMatrix(std::string& output, const math::Matrix* matrix, const PrintArgs& args)
{
  const floatt zrr = args.zrr;
  const bool repeats = args.repeats;
  const std::string sectionSeparator = args.sectionSeparator;

  if (matrix == NULL)
  {
    output = "nullptr";
    return;
  }
  MatrixRange matrixRange (matrix);
  PrintMatrix(output, matrixRange, args);
}
}
