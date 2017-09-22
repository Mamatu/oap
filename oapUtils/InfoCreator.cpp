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

#include "InfoCreator.h"
#include "MatrixUtils.h"

InfoCreator::InfoCreator() : m_expected(NULL), m_output(NULL) {}

InfoCreator::~InfoCreator() {}

void InfoCreator::onSetInfoTypeCallback(const InfoType& infoType) {}

void InfoCreator::onSetExpectedCallback(math::Matrix* expected) {}

void InfoCreator::onSetOutputCallback(math::Matrix* output) {}

math::Matrix* InfoCreator::getExpectedMatrix() const { return m_expected; }

math::Matrix* InfoCreator::getOutputMatrix() const { return m_output; }

void InfoCreator::setExpected(math::Matrix* expected) {
  m_expected = expected;
  onSetExpectedCallback(m_expected);
}

void InfoCreator::setOutput(math::Matrix* output) {
  m_output = output;
  onSetOutputCallback(m_output);
}

void InfoCreator::setInfoType(const InfoType& infoType) {
  m_infoType = infoType;
}

void InfoCreator::getInfo(std::string& output) const { output = m_info; }

void InfoCreator::getSLInfo(std::string& outputStr, const std::string& label,
                            ICMethod methodre, ICMethod methodim,
                            math::Matrix* diffmatrix) const {
  std::stringstream sstream;
  if (isRe(diffmatrix)) {
    std::pair<floatt, uintt> largestre = (this->*methodre)(diffmatrix);
    sstream << largestre.first;
    outputStr += "Diff " + label + "  = (" + sstream.str() + ",";
    sstream.str("");
  } else {
    outputStr += "Diff " + label + " = (null,";
  }
  if (isIm(diffmatrix)) {
    std::pair<floatt, uintt> largestim = (this->*methodim)(diffmatrix);
    sstream << largestim.first;
    outputStr += sstream.str() + ")";
    sstream.str("");
  } else {
    outputStr += "null)";
  }
  outputStr += "\n";
}

void InfoCreator::createInfo(std::string& outputStr, const InfoType& infoType,
                             math::Matrix* diffmatrix) const {
  math::Matrix* output = getOutputMatrix();
  math::Matrix* expected = getExpectedMatrix();
  bool isequal = diffmatrix == NULL;
  if (!isequal) {
    if (infoType.getInfo() & InfoType::MEAN) {
      printMeans(outputStr, diffmatrix);
    }
    if (infoType.getInfo() & InfoType::LARGEST_DIFF) {
      getSLInfo(outputStr, "largest", &InfoCreator::getLargestReValue,
                &InfoCreator::getLargestImValue, diffmatrix);
    }
    if (infoType.getInfo() & InfoType::SMALLEST_DIFF) {
      getSLInfo(outputStr, "smallest", &InfoCreator::getSmallestReValue,
                &InfoCreator::getSmallestImValue, diffmatrix);
    }
    if (infoType.getInfo() & InfoType::ELEMENTS) {
      outputStr += "\n";
      printMatrix(outputStr, "Output = ", m_output);
      printMatrix(outputStr, "Expected = ", m_expected);
      printMatrix(outputStr, "Diff = ", diffmatrix);
      outputStr += "\n";
    }
  }
}

void InfoCreator::printMatrix(std::string& output, const std::string& message,
                              math::Matrix* matrix) const {
  std::string matrixStr;
  getString(matrixStr, matrix);
  output += message + matrixStr + "\n";
}

bool InfoCreator::printMeans(std::string& output,
                             math::Matrix* diffmatrix) const {
  math::Matrix* expectedMatrix = getExpectedMatrix();
  math::Matrix* outputMatrix = getOutputMatrix();
  if (expectedMatrix == NULL || outputMatrix == NULL) {
    return false;
  }
  math::Matrix* diff = diffmatrix;
  printMean(output, "Expected mean = ", expectedMatrix);
  printMean(output, "Output mean = ", outputMatrix);
  printMean(output, "Diff mean = ", diff);
  return true;
}

void InfoCreator::printMean(std::string& output, const std::string& message,
                            math::Matrix* matrix) const {
  std::string matrixStr;
  floatt are = 0;
  floatt aim = 0;
  getMean(are, aim, matrix);
  std::stringstream sstream1;
  std::stringstream sstream2;
  sstream1 << are;
  sstream2 << aim;
  output += message + "(" + sstream1.str() + "," + sstream2.str() + ") ";
}

bool InfoCreator::isEqual() {
  math::Matrix* output = getOutputMatrix();
  math::Matrix* expected = getExpectedMatrix();
  math::Matrix* diffMatrix = NULL;
  bool result = compare(output, expected, &diffMatrix);
  if (result == false) {
    createInfo(m_info, m_infoType, diffMatrix);
    destroyMatrix(diffMatrix);
  }
  return result;
}

bool InfoCreator::hasValues() {
  math::Matrix* output = getOutputMatrix();
  math::Matrix* expected = getExpectedMatrix();
  math::Matrix* diffMatrix = NULL;
  bool result = compareValues(output, expected, &diffMatrix);
  if (diffMatrix != NULL) {
    createInfo(m_info, m_infoType, diffMatrix);
    destroyMatrix(diffMatrix);
  }
  return result;
}
