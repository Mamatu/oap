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

#ifndef OAP_NEURAL_NETWORK_STRUCTURE_H
#define OAP_NEURAL_NETWORK_STRUCTURE_H

#include <map>
#include <vector>

#include "oapLayerStructure.h"

template <typename T>
class NetworkS
{
  public:
    virtual ~NetworkS()
    {}

    void setExpected (math::Matrix* expected, ArgType argType, FPHandler handler = 0);
    math::Matrix* getExpected (FPHandler handler = 0) const;

  protected:
    std::vector<floatt> m_errorsVec;

    floatt m_learningRate = 0.1f;
    uintt m_step = 1;

    using ExpectedOutputs = std::map<FPHandler, T>;
    ExpectedOutputs m_expectedOutputs;

    virtual void setExpectedProtected (typename ExpectedOutputs::mapped_type& holder, math::Matrix* expected, ArgType argType) = 0;
    virtual math::Matrix* convertExpectedProtected (T t) const = 0;
};

template<typename T>
void NetworkS<T>::setExpected (math::Matrix* expected, ArgType argType, FPHandler handler)
{
  typename ExpectedOutputs::mapped_type& holder = m_expectedOutputs[handler];
  
  setExpectedProtected (holder, expected, argType);
}

template<typename T>
math::Matrix* NetworkS<T>::getExpected (FPHandler handler) const
{
  auto it = m_expectedOutputs.find (handler);
  if (it == m_expectedOutputs.end ())
  {
    return nullptr;
  }
  return convertExpectedProtected (it->second);
}

#endif
