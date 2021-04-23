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

#include "oapNetworkTopology.h"

namespace oap
{

uintt InputTopology::Data::getMStructsCount() const
{
  return mhandlers.size();
}

uintt InputTopology::Data::getInputLayersCount() const
{
  return lhandlers.size();
}

InputTopology::MStruct InputTopology::Data::getMStructByIdx (uintt idx) const
{
  oapAssert (idx < mhandlers.size());
  return mhandlers[idx];
}

InputTopology::MStruct InputTopology::Data::getMStructByHandler (MHandler handler) const
{
  oapAssert (handler < mhandlers.size());
  return mhandlers[handler];
}

std::vector<MHandler> InputTopology::Data::getInputLayer (uintt idx) const
{
  oapAssert (idx < lhandlers.size());
  return lhandlers[idx];
}

std::vector<InputTopology::MStruct> InputTopology::Data::getInputLayerMatrices (uintt idx) const
{
  const auto& mhandlers = getInputLayer(idx);

  std::vector<MStruct> output;

  for (const auto& handler : mhandlers)
  {
    output.push_back (getMStructByHandler (handler));
  }
  return output;
}

LHandler InputTopology::addVertical(const math::MatrixInfo& minfo, uintt count)
{
  oapAssert (minfo.columns() == 1);

  MHandler mhandler = addMatrix (minfo, m_data.lhandlers.size());
  std::vector<MHandler> mhandlers (count, mhandler);

  m_data.lhandlers.push_back (mhandlers);
  return m_data.lhandlers.size() - 1;
}

LHandler InputTopology::addVertical(uintt rows, uintt count)
{
  math::MatrixInfo minfo (1, rows, true, false);
  return addVertical (minfo, count);
}

LHandler InputTopology::addSharedVertical(const math::MatrixInfo& minfo, uintt count, MHandler shandler)
{
  oapAssert (shandler < m_data.mhandlers.size());
  const auto& m = m_data.mhandlers[shandler];

  oapAssert (minfo.columns () == m.minfo.columns());
  oapAssert (m.minfo.rows() % minfo.rows() == 0);

  std::vector<MHandler> mhandlers;

  for (uintt idx = 0; idx < count; ++idx)
  {
    oap::math::MatrixLoc loc = {0, minfo.rows() * idx};
    oap::math::MatrixDim dim = {minfo.columns(), minfo.rows()};
    MHandler mhandler = addSharedMatrix (minfo, m_data.lhandlers.size(), shandler, loc, dim);
    mhandlers.push_back (mhandler);
  }

  m_data.lhandlers.push_back (mhandlers);
  return m_data.lhandlers.size() - 1;
}

LHandler InputTopology::addSharedVertical(uintt rows, uintt count, MHandler mhandler)
{
  math::MatrixInfo minfo (1, rows, true, false);
  return addSharedVertical (minfo, count, mhandler);
}

oap::math::Matrix InputTopology::getMatrixRef(const math::MatrixInfo& minfo)
{
  math::Matrix matrix;
  matrix.re.dims = {0, 0};
  matrix.im.dims = {0, 0};

  if (minfo.isRe)
  {
    matrix.re.dims = {minfo.columns(), minfo.rows()};
    matrix.re.ptr = nullptr;
  }

  if (minfo.isIm)
  {
    matrix.im.dims = {minfo.columns(), minfo.rows()};
    matrix.im.ptr = nullptr;
  }

  return matrix;  
}

MHandler InputTopology::addMatrix(const math::MatrixInfo& minfo, LHandler lhandler)
{
  oapAssert (lhandler < m_data.lhandlers.size());
  auto& inputLayer = m_data.lhandlers[lhandler];

  MStruct mstruct = {minfo, 0, {0, 0}, {0, 0}};
  m_data.mhandlers.push_back (mstruct);

  MHandler mhandler = m_data.mhandlers.size() - 1;

  inputLayer.push_back (mhandler);
  return mhandler;
}

MHandler InputTopology::addSharedMatrix(const math::MatrixInfo& minfo, LHandler lhandler, MHandler sharingMatrix, const oap::math::MatrixLoc& loc, const oap::math::MatrixDim& dim)
{
  oapAssert (lhandler < m_data.lhandlers.size());
  auto& inputLayer = m_data.lhandlers[lhandler];
  oapAssert (sharingMatrix < m_data.mhandlers.size());

  MStruct mstruct = {minfo, sharingMatrix, loc, dim};
  m_data.mhandlers.push_back (mstruct);

  MHandler mhandler = m_data.mhandlers.size() - 1;

  inputLayer.push_back (mhandler);
  return mhandler;
}

const InputTopology::Data& InputTopology::getData () const
{
  return m_data;
}

}
