#include "PatternsClassificationHost.h"

namespace oap
{
  PatternsClassificationParser::PatternsClassificationParser ()
  {
    m_parser.registerArg ("pattern1", [this](const std::string& value)
    {
      m_args.patternPath1 = value;
      debug ("patternPath1 = %s", m_args.patternPath1.c_str ());
    });

    m_parser.registerArg ("pattern2", [this](const std::string& value)
    {
      m_args.patternPath2 = value;
      debug ("patternPath2 = %s", m_args.patternPath2.c_str ());
    });

    auto errorTypeF = [this](const std::string& value)
    {
      std::array<std::string, 4> expected =
      {
        "cross_entropy", "ce",
        "mean_square_error", "mse"
      };
      if (value == expected[0] || value == expected[1])
      {
        m_args.errorType = oap::ErrorType::CROSS_ENTROPY;
      }
      else if (value == expected[2] || value == expected[3])
      {
        m_args.errorType = oap::ErrorType::MEAN_SQUARE_ERROR;
      }
      else
      {
        std::stringstream sstream;
        sstream << "Not expected value of argument \"" << value << "\" (should be";
        for (auto it = expected.cbegin(); it != expected.cend(); ++it)
        {
          sstream << " \"" << *it << "\"";
        }
        sstream << ")";
        throw std::runtime_error (sstream.str ());
      }
    };

    m_parser.registerArg ("error_type", errorTypeF);

    auto layersF = [this](const std::string& value)
    {
      try
      {
        m_args.networkLayers.clear ();

        std::string numbers = "0123456789";
        size_t pos = 0;
        size_t pos1 = 0;

        auto addNeuronsCount = [this](const std::string& value, size_t pos, size_t pos1)
        {
          std::string sub = value.substr (pos, pos1 - pos);
          debug("%s", sub.c_str());
          uintt neuronsCount = static_cast<uintt>(std::stoi (sub));
          m_args.networkLayers.push_back (neuronsCount);
        };

        debug("Layers:");
        while ((pos1 = value.find_first_not_of (numbers, pos)) != std::string::npos)
        {
          addNeuronsCount (value, pos, pos1);

          pos = value.find_first_of (numbers, pos1);
        }
        addNeuronsCount (value, pos, value.length ());
      }
      catch (const std::exception& exception)
      {
        std::stringstream sstream;
        sstream << "Exception during parsing neurons count in layers: \"" << exception.what() << "\"";
        throw std::runtime_error (sstream.str ());
      }
    };
  
    m_parser.registerArg ("layers", layersF);

    auto savingF = [this](const std::string& path)
    {
      m_args.savingPath = path;
    };
    auto loadingF = [this](const std::string& path)
    {
      m_args.loadingPath = path;
    };

    m_parser.registerArg ("save", savingF);
    m_parser.registerArg ("load", loadingF);
  }

  void PatternsClassificationParser::parse (int argc, char **argv) const
  {
    m_parser.parse (argc, argv);
  }
}
