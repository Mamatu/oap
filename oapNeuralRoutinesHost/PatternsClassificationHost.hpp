/*
 * Copyright 2016 - 2021 Marcin Matula
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

#ifndef OAP_PATTERNS_CLASSIFICATION_ARGS_H
#define OAP_PATTERNS_CLASSIFICATION_ARGS_H

#include "ArgsParser.hpp"
#include "GraphicUtils.hpp"
#include "oapLayerStructure.hpp"
#include "Math.hpp"

namespace oap
{
  class PatternsClassificationParser : public IArgsParser
  {
    public:
      struct Args
      {
        std::string patternPath1 = "oapNeural/data/text/a.png";
        std::string patternPath2 = "oapNeural/data/text/b.png";
      
        std::string loadingPath = "";
        std::string savingPath = "";

        oap::ErrorType errorType = oap::ErrorType::CROSS_ENTROPY;

        std::vector<uint> networkLayers;
      
        using OutputCallback = std::function<void(const std::vector<floatt>&)>;
      
        OutputCallback m_onOutput1;
        OutputCallback m_onOutput2;
        std::function<void(const oap::ImageSection&, const oap::ImageSection&, bool)> m_onOpenFile;
      };

      PatternsClassificationParser ();

      PatternsClassificationParser (const PatternsClassificationParser&) = delete;
      PatternsClassificationParser (PatternsClassificationParser&&) = delete;
      PatternsClassificationParser& operator= (const PatternsClassificationParser&) = delete;
      PatternsClassificationParser& operator= (PatternsClassificationParser&&) = delete;

      virtual void parse (int argc, char* const* argv) const override;

      Args getArgs() const
      {
        return m_args;
      }

    private:
      Args m_args;
      ArgsParser m_parser;
  };
}

#endif
