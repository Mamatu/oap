#ifndef INFOCREATOR_H
#define INFOCREATOR_H

#include "MatrixAPI.h"
#include "InfoType.h"
#include "Utils.h"

class InfoCreator {
 private:
  math::Matrix* m_expected;
  math::Matrix* m_output;
  InfoType m_infoType;
  bool m_isEqual;

 protected:
  inline void printMatrix(std::string& output, const std::string& message,
                          math::Matrix* matrix) const;

  inline void printMean(std::string& output, const std::string& message,
                        math::Matrix* matrix) const;

  inline floatt calculateMean(const matrixUtils::MatrixRange& matrixRange,
                              floatt (*GetValue)(const math::Matrix*, uintt,
                                                 uintt)) const;

 public:
  inline InfoCreator();

  inline InfoCreator(math::Matrix* expected, math::Matrix* output,
                     const InfoType& infoType);

  inline virtual ~InfoCreator();

  inline void setExpected(math::Matrix* expected);
  inline void setOutput(math::Matrix* output);
  inline void setInfoType(const InfoType& infoType);

  inline void printInfo(std::string& info);

  inline bool isEquals() const;
};

#include "MatrixUtils.h"

InfoCreator::InfoCreator()
    : m_expected(NULL), m_output(NULL), m_infoType(InfoType()) {}

InfoCreator::InfoCreator(math::Matrix* expected, math::Matrix* output,
                         const InfoType& infoType)
    : m_expected(expected), m_output(output), m_infoType(infoType) {}

InfoCreator::~InfoCreator() {}

void InfoCreator::setExpected(math::Matrix* expected) { m_expected = expected; }

void InfoCreator::setOutput(math::Matrix* output) { m_output = output; }

void InfoCreator::setInfoType(const InfoType& infoType) {
  m_infoType = infoType;
}

void InfoCreator::printMatrix(std::string& output, const std::string& message,
                              math::Matrix* matrix) const {
  std::string matrixStr;
  matrixUtils::MatrixRange matrixRange(matrix, m_infoType);
  matrixUtils::PrintMatrix(matrixStr, matrixRange);
  output += message + matrixStr;
}

floatt InfoCreator::calculateMean(const matrixUtils::MatrixRange& matrixRange,
                                  floatt (*GetValue)(const math::Matrix*, uintt,
                                                     uintt)) const {
  floatt value = 0;
  uintt rows = matrixRange.getERow();
  uintt columns = matrixRange.getEColumn();
  uintt length = columns * rows;
  for (uintt fa = matrixRange.getBColumn(); fa < columns; ++fa) {
    for (uintt fb = matrixRange.getBRow(); fb < rows; ++fb) {
      value += (*GetValue)(matrixRange.getMatrix(), fa, fb);
    }
  }
  return value / static_cast<floatt>(length);
}

void InfoCreator::printMean(std::string& output, const std::string& message,
                            math::Matrix* matrix) const {
  std::string matrixStr;
  floatt are = 0;
  floatt aim = 0;
  matrixUtils::MatrixRange matrixRange(matrix);

  if (matrixRange.isReValues()) {
    are = calculateMean(matrixRange, GetRe);
  }

  if (matrixRange.isImValues()) {
    aim = calculateMean(matrixRange, GetIm);
  }

  std::stringstream sstream1;
  std::stringstream sstream2;
  sstream1 << are;
  sstream2 << aim;
  output += message + "(" + sstream1.str() + "," + sstream2.str() + ") ";
}

void InfoCreator::printInfo(std::string& info) {
  math::Matrix* diffmatrix = NULL;
  bool isequal = utils::IsEqual((*m_expected), (*m_output), &diffmatrix);
  if (!isequal) {
    if (m_infoType.getInfo() & InfoType::MEAN) {
      printMean(info, "Expected mean = ", m_expected);
      printMean(info, "Output mean = ", m_output);
      printMean(info, "Diff mean = ", diffmatrix);
      info += "\n";
    }
    if (m_infoType.getInfo() & InfoType::ELEMENTS) {
      printMatrix(info, "Output = ", m_output);
      printMatrix(info, "Diff = ", diffmatrix);
      info += "\n";
    }
  }
  host::DeleteMatrix(diffmatrix);
  m_isEqual = isequal;
}

bool InfoCreator::isEquals() const { return m_isEqual; }

#endif  // INFOCREATOR_H
