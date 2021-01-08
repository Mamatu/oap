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

#include "PointsClassification_Test.h"
#include "oapCudaMatrixUtils.h"
#include "CuProceduresApi.h"
#include "MultiMatricesCuProcedures.h"
#include "oapNetworkCudaApi.h"

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

  template<typename T, typename Data>
  std::vector<T> convertToVec (const Data& data)
  {
    std::vector<T> entries;
    for (const auto& d : data)
    {
      entries.emplace_back (d);
    }
    return entries;
  };

};

TEST_F(OapClassificationTests, RingDataTest)
{
  auto* singleApi = new oap::CuProceduresApi();
  auto* multiApi = new oap::MultiMatricesCuProcedures(singleApi);
  auto* nca = new oap::NetworkCudaApi();
  oap::runPointsClassification (123456789, singleApi, multiApi, nca, oap::cuda::CopyHostMatrixToDeviceMatrix, oap::cuda::GetMatrixInfo);
}
