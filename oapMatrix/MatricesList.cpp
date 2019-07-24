/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "MatricesList.h"

MatricesList::MatricesList (const std::string& id) : m_id(id)
{}

MatricesList::~MatricesList ()
{
  checkOnDelete ();
}

const MatricesList::MatrixInfos& MatricesList::getAllocated() const
{
  return m_matrixInfos;
}

void MatricesList::add (math::Matrix* matrix, const math::MatrixInfo& minfo)
{
  m_matrixInfos[matrix] = minfo;

  MatrixInfos::iterator it = m_deletedMatrixInfos.find (matrix);
  if (it != m_deletedMatrixInfos.end ())
  {
    m_deletedMatrixInfos.erase (it);
  }

  auto size = minfo.getSize ();
  //debugInfo ("Allocate in %s: matrix = %p %s", m_id.c_str(), matrix, minfo.toString().c_str());
}

math::MatrixInfo MatricesList::getMatrixInfo (const math::Matrix* matrix) const
{
  const auto& map = getAllocated();
  auto it = map.find (matrix);

  if (it == map.end())
  {
    std::stringstream stream;
    stream << "Matrix " << matrix << " does not exist or was not allocated in proper way.";
    throw std::runtime_error (stream.str ());
  }

  return it->second;
}

bool MatricesList::contains (const math::Matrix* matrix) const
{
  const auto& map = getAllocated();
  auto it = map.find (matrix);

	return (it != map.end());
}

math::MatrixInfo MatricesList::remove (const math::Matrix* matrix)
{
  math::MatrixInfo minfo;

  MatrixInfos::iterator it = m_matrixInfos.find(matrix);
  if (m_matrixInfos.end() != it)
  {
    m_deletedMatrixInfos[matrix] = it->second;
    minfo = it->second;

    m_matrixInfos.erase(it);
  }
  else
  {
    MatrixInfos::iterator it = m_deletedMatrixInfos.find(matrix);
    if (it != m_deletedMatrixInfos.end ())
    {
      debugError ("Double deallocation in %s: matrix = %p %s", m_id.c_str(), matrix, it->second.toString().c_str());
      debugAssert (false);
    }
    else
    {
      debugError ("Not found in %s: matrix = %p", m_id.c_str(), matrix);
      debugAssert (false);
    }
  }
  return minfo;
}


void MatricesList::checkOnDelete()
{
  if (m_matrixInfos.size() > 0)
  {
    debugError ("Memleak: not deallocated matrices");
    for (MatrixInfos::iterator it = m_matrixInfos.begin(); it != m_matrixInfos.end(); ++it)
    {
      debug("Memleak in %s: matrix = %p %s not deallocated", m_id.c_str(), it->first, it->second.toString().c_str());
    }
    //debugAssert (false);
  }
}
