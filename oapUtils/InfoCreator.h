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

#ifndef INFOCREATOR_H
#define INFOCREATOR_H

#include "MatrixAPI.h"
#include "InfoType.h"

class InfoCreator;

typedef std::pair<floatt, uintt> (InfoCreator::*ICMethod)(math::Matrix*) const;

class InfoCreator {
 private:
  math::Matrix* m_expected;
  math::Matrix* m_output;

  std::string m_info;
  InfoType m_infoType;

 public:
  InfoCreator();
  virtual ~InfoCreator();

  void setExpected(math::Matrix* expected);
  void setOutput(math::Matrix* output);

  void setInfoType(const InfoType& infoType);
  void getInfo(std::string& output) const;

 protected:
  class ComplexInternal : public Complex {
   public:
    bool m_isValid;
    ComplexInternal(bool isValid);
  };

  virtual math::Matrix* getExpectedMatrix() const;
  virtual math::Matrix* getOutputMatrix() const;

  virtual void onSetInfoTypeCallback(const InfoType& infoType);
  virtual void onSetExpectedCallback(math::Matrix* expected);
  virtual void onSetOutputCallback(math::Matrix* output);

  virtual void getString(std::string& output, math::Matrix* matrix) const = 0;

  virtual void getMean(floatt& re, floatt& im, math::Matrix* matrix) const = 0;

  virtual bool compare(math::Matrix* matrix1, math::Matrix* matrix2,
                       math::Matrix** diffMatrix) const = 0;

  virtual bool compareValues(math::Matrix* matrix1, math::Matrix* matrix2,
                             math::Matrix** diffMatrix) const = 0;

  virtual void destroyMatrix(math::Matrix* diffMatrix) const = 0;

  virtual bool isRe(math::Matrix* matrix) const = 0;

  virtual bool isIm(math::Matrix* matrix) const = 0;
  virtual std::pair<floatt, uintt> getLargestReValue(
      math::Matrix* matrix) const = 0;
  virtual std::pair<floatt, uintt> getLargestImValue(
      math::Matrix* matrix) const = 0;

  virtual std::pair<floatt, uintt> getSmallestReValue(
      math::Matrix* matrix) const = 0;
  virtual std::pair<floatt, uintt> getSmallestImValue(
      math::Matrix* matrix) const = 0;

 private:
  void getSLInfo(std::string& outputStr, const std::string& label,
                 ICMethod methodre, ICMethod methodim,
                 math::Matrix* diffmatrix) const;

  void createInfo(std::string& output, const InfoType& infoType,
                  math::Matrix* diffmatrix) const;

 public:
  void printMatrix(std::string& output, const std::string& message,
                   math::Matrix* matrix) const;

  bool printMeans(std::string& output, math::Matrix* diffmatrix) const;

  void printMean(std::string& output, const std::string& message,
                 math::Matrix* matrix) const;

  bool isEqual();
  bool hasValues();
};

#endif  // INFOCREATOR_H
