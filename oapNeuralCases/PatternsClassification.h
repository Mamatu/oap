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

#ifndef OAP_PATTERNS_CLASSIFICATION_H
#define OAP_PATTERNS_CLASSIFICATION_H

#include <string>
#include <functional>

#include "oapNetwork.h"
#include "PngFile.h"

namespace oap
{
namespace classification
{

struct Args
{
  std::string patternPath1 = "oapNeural/data/text/a.png";
  std::string patternPath2 = "oapNeural/data/text/b.png";

  Network::ErrorType errorTpe = Network::ErrorType::CROSS_ENTROPY;

  std::vector<uint> networkLayers = {20*20, 20, 1};

  using OutputCallback = std::function<void(const std::vector<floatt>&)>;

  OutputCallback m_onOutput1;
  OutputCallback m_onOutput2;
  std::function<void(const oap::OptSize&, const oap::OptSize&, bool)> m_onOpenFile;
};

int run_PatternsClassification (int argc, char **argv);

int run_PatternsClassification (const Args& args);

int run_PatternsClassification ();

}
}

#endif
