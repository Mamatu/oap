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

#ifndef INFOCREATOR_H
#define INFOCREATOR_H

#include "MatrixAPI.h"
#include "InfoType.h"

class InfoCreator;

typedef std::pair<floatt, uintt> (InfoCreator::*ICMethod)(math::ComplexMatrix*) const;

class InfoCreator {
 private:
  math::ComplexMatrix* m_expected;
  math::ComplexMatrix* m_output;

  std::string m_info;
  InfoType m_infoType;

 public:
  InfoCreator();
  virtual ~InfoCreator();

  void setExpected(math::ComplexMatrix* expected);
  void setOutput(math::ComplexMatrix* output);

  void setInfoType(const InfoType& infoType);
  void getInfo(std::string& output) const;

 protected:
  class ComplexInternal : public Complex {
   public:
    bool m_isValid;
    ComplexInternal(bool isValid);
  };

  virtual math::ComplexMatrix* getExpectedMatrix() const;
  virtual math::ComplexMatrix* getOutputMatrix() const;

  virtual void onSetInfoTypeCallback(const InfoType& infoType);
  virtual void onSetExpectedCallback(math::ComplexMatrix* expected);
  virtual void onSetOutputCallback(math::ComplexMatrix* output);

  virtual void getString(std::string& output, math::ComplexMatrix* matrix) const = 0;

  virtual void getMean(floatt& re, floatt& im, math::ComplexMatrix* matrix) const = 0;

  virtual bool compare(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                       floatt tolerance, math::ComplexMatrix** diffMatrix) const = 0;

  virtual bool compareValues(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                             floatt tolerance, math::ComplexMatrix** diffMatrix) const = 0;

  virtual void destroyMatrix(math::ComplexMatrix* diffMatrix) const = 0;

  virtual bool isRe(math::ComplexMatrix* matrix) const = 0;

  virtual bool isIm(math::ComplexMatrix* matrix) const = 0;
  virtual std::pair<floatt, uintt> getLargestReValue(
      math::ComplexMatrix* matrix) const = 0;
  virtual std::pair<floatt, uintt> getLargestImValue(
      math::ComplexMatrix* matrix) const = 0;

  virtual std::pair<floatt, uintt> getSmallestReValue(
      math::ComplexMatrix* matrix) const = 0;
  virtual std::pair<floatt, uintt> getSmallestImValue(
      math::ComplexMatrix* matrix) const = 0;

 private:
  void getSLInfo(std::string& outputStr, const std::string& label,
                 ICMethod methodre, ICMethod methodim,
                 math::ComplexMatrix* diffmatrix) const;

  void createInfo(std::string& output, const InfoType& infoType,
                  math::ComplexMatrix* diffmatrix) const;

 public:
  void printMatrix(std::string& output, const std::string& message,
                   math::ComplexMatrix* matrix) const;

  bool printMeans(std::string& output, math::ComplexMatrix* diffmatrix) const;

  void printMean(std::string& output, const std::string& message,
                 math::ComplexMatrix* matrix) const;

  bool isEqual(floatt tolerance);
  bool hasValues(floatt tolerance);
};

#endif  // INFOCREATOR_H
