/*
 * Copyright 2016, 2017 Marcin Matula
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

#include "DataLoader.h"
#include "PngFile.h"

#include "MainAPExecutor.h"
#include "MainFourierExecutor.h"

int main()
{
  oap::cuda::Context::Instance().create();

  oap::MainAPExecutor mainExecutor;
  oap::MainFourierExecutor mainFExec;

  size_t wantedCount = 5;

  mainExecutor.setMaxIterationCounter(10);
  mainExecutor.setEigensType(ArnUtils::HOST);
  mainExecutor.setWantedCount (wantedCount);
  mainExecutor.setInfo (oap::DataLoader::Info("oap2dt3d/data/images_monkey_125", "image_", 125, true));

  std::shared_ptr<oap::Outcome> outcome = mainExecutor.run ();

  for (size_t idx = 0; idx < wantedCount; ++idx)
  {
    floatt value = outcome->getValue(idx);
    floatt error = outcome->getError(idx);
    const math::Matrix* vec = outcome->getVector(idx);
    printf("---------------------------------------\n");
    printf("wanted = %f \n", value);
    printf("error = %f \n", error);
    oap::host::PrintMatrix("wantedEV = ", vec);
    printf("---------------------------------------\n");

  }

  mainFExec.setOutcome (outcome);
  mainFExec.run();

  oap::cuda::Context::Instance().destroy();
  return 0;
}
