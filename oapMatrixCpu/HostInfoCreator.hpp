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

#ifndef HOSTINFOCREATOR_H
#define HOSTINFOCREATOR_H

#include "InfoCreator.hpp"
#include "HostProcedures.hpp"

class HostInfoCreator : public InfoCreator {
 private:
 protected:
  virtual void setInfoTypeCallback(const InfoType& infoType);
  virtual void setExpectedCallback(math::ComplexMatrix* expected);
  virtual void setOutputCallback(math::ComplexMatrix* output);

  virtual void getString(std::string& output, math::ComplexMatrix* matrix) const;

  virtual void getMean(floatt& re, floatt& im, math::ComplexMatrix* matrix) const;

  virtual bool compare(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                       floatt tolerance, math::ComplexMatrix** diffMatrix) const;

  virtual bool compareValues(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                             floatt tolerance, math::ComplexMatrix** diffMatrix) const;

  virtual void destroyMatrix(math::ComplexMatrix* diffMatrix) const;

  virtual std::pair<floatt, uintt> getLargestReValue(
      math::ComplexMatrix* matrix) const;
  virtual std::pair<floatt, uintt> getLargestImValue(
      math::ComplexMatrix* matrix) const;

  virtual std::pair<floatt, uintt> getSmallestReValue(
      math::ComplexMatrix* matrix) const;
  virtual std::pair<floatt, uintt> getSmallestImValue(
      math::ComplexMatrix* matrix) const;

  virtual bool isRe(math::ComplexMatrix* matrix) const;

  virtual bool isIm(math::ComplexMatrix* matrix) const;
};

#endif  // INFOCREATORHOST
