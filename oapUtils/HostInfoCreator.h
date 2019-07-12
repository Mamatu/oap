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

#ifndef HOSTINFOCREATOR_H
#define HOSTINFOCREATOR_H

#include "InfoCreator.h"
#include "HostProcedures.h"

class HostInfoCreator : public InfoCreator {
 private:
 protected:
  virtual void setInfoTypeCallback(const InfoType& infoType);
  virtual void setExpectedCallback(math::Matrix* expected);
  virtual void setOutputCallback(math::Matrix* output);

  virtual void getString(std::string& output, math::Matrix* matrix) const;

  virtual void getMean(floatt& re, floatt& im, math::Matrix* matrix) const;

  virtual bool compare(math::Matrix* matrix1, math::Matrix* matrix2,
                       math::Matrix** diffMatrix) const;

  virtual bool compareValues(math::Matrix* matrix1, math::Matrix* matrix2,
                             math::Matrix** diffMatrix) const;

  virtual void destroyMatrix(math::Matrix* diffMatrix) const;

  virtual std::pair<floatt, uintt> getLargestReValue(
      math::Matrix* matrix) const;
  virtual std::pair<floatt, uintt> getLargestImValue(
      math::Matrix* matrix) const;

  virtual std::pair<floatt, uintt> getSmallestReValue(
      math::Matrix* matrix) const;
  virtual std::pair<floatt, uintt> getSmallestImValue(
      math::Matrix* matrix) const;

  virtual bool isRe(math::Matrix* matrix) const;

  virtual bool isIm(math::Matrix* matrix) const;
};

#endif  // INFOCREATORHOST
