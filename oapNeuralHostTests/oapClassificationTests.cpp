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

#include "PointsClassification_Test.hpp"
#include "oapHostComplexMatrixApi.hpp"

#include "HostProcedures.hpp"
#include "MultiMatricesHostProcedures.hpp"
#include "oapNetworkHostApi.hpp"

class OapClassificationTests : public testing::Test
{
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(OapClassificationTests, RingDataTest)
{
  auto* singleApi = new oap::HostProcedures ();
  oap::MultiMatricesHostProcedures* multiApi = new oap::MultiMatricesHostProcedures (singleApi);
  auto* nha = new oap::NetworkHostApi ();
  oap::runPointsClassification (123456789, singleApi, multiApi, nha, oap::chost::CopyHostMatrixToHostMatrix, oap::chost::GetMatrixInfo);

  delete singleApi;
  delete multiApi;
  delete nha;
}

TEST_F(OapClassificationTests, RingDataTest_MultiMatrices)
{
  auto* singleApi = new oap::HostProcedures ();
  oap::MultiMatricesHostProcedures* multiApi = new oap::MultiMatricesHostProcedures (singleApi);
  auto* nha = new oap::NetworkHostApi ();
  oap::runPointsClassification_multiMatrices (123456789, singleApi, multiApi, nha, oap::chost::CopyHostMatrixToHostMatrix, oap::chost::GetMatrixInfo);

  delete singleApi;
  delete multiApi;
  delete nha;
}

