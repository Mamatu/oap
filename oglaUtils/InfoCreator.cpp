#include "InfoCreator.h"
#include "MatrixUtils.h"

InfoCreator::InfoCreator() {}

InfoCreator::~InfoCreator() {}

void InfoCreator::setInfoType(const InfoType& infoType) {
  m_infoType = infoType;
  setInfoTypeCallback(m_infoType);
}

InfoType InfoCreator::getInfoType() const { return m_infoType; }

void InfoCreator::setInfoTypeCallback(const InfoType& infoType) {}

void InfoCreator::setExpectedCallback(math::Matrix* expected) {}

void InfoCreator::setOutputCallback(math::Matrix* output) {}

math::Matrix* InfoCreator::getExpectedMatrix() const { return m_expected; }

math::Matrix* InfoCreator::getOutputMatrix() const { return m_output; }

void InfoCreator::setExpected(math::Matrix* expected) {
  m_expected = expected;
  setExpectedCallback(m_expected);
}

void InfoCreator::setOutput(math::Matrix* output) {
  m_output = output;
  setOutputCallback(m_output);
}

void InfoCreator::getInfo(std::string& output) const {
  math::Matrix* diffmatrix = NULL;
  math::Matrix* diff =
      createDiffMatrix(m_expected, m_output, InfoCreator::HOST);
  bool isequal = diff == NULL;
  if (!isequal) {
    if (m_infoType.getInfo() & InfoType::MEAN) {
      if (printMean(output, output, InfoCreator::HOST) == false) {
        printMean(output, output, InfoCreator::DEVICE);
      }
      output += "\n";
    }
    if (m_infoType.getInfo() & InfoType::ELEMENTS) {
      printMatrix(output, "Output = ", m_output);
      printMatrix(output, "Diff = ", diffmatrix);
      output += "\n";
    }
  }
  destroyDiffMatrix(diff);
  diff = NULL;
}

void InfoCreator::printMatrix(std::string& output, const std::string& message,
                              math::Matrix* matrix) const {
  std::string matrixStr;
  matrixUtils::MatrixRange matrixRange(matrix, m_infoType);
  matrixUtils::PrintMatrix(matrixStr, matrixRange);
  output += message + matrixStr;
}

bool InfoCreator::printMean(std::string& output, const std::string& message,
                            InfoCreator::MatrixType matrixType) const {
  InfoCreator::MatrixType mt1 =
      static_cast<InfoCreator::MatrixType>(matrixType | InfoCreator::EXPECTED);
  InfoCreator::MatrixType mt2 =
      static_cast<InfoCreator::MatrixType>(matrixType | InfoCreator::OUTPUT);
  math::Matrix* expectedMatrix = getMatrix(mt1);
  math::Matrix* outputMatrix = getMatrix(mt2);
  if (expectedMatrix == NULL || outputMatrix == NULL) {
    return false;
  }
  printMean(output, "Expected mean = ", expectedMatrix);
  printMean(output, "Output mean = ", outputMatrix);
  return true;
  // printMean(output, "Diff mean = ", diffmatrix);
}

void InfoCreator::printMean(std::string& output, const std::string& message,
                            math::Matrix* matrix) const {
  std::string matrixStr;
  floatt are = 0;
  floatt aim = 0;
  /*math::Matrix expectedHost =
      getMatrix(InfoCreator::HOST | InfoCreator::EXPECTED);
  math::Matrix outputHost = getMatrix(InfoCreator::HOST | InfoCreator::OUTPUT);
  if (matrixRange.isReValues()) {
    are = getMean(matrix, InfoCreator);
  }
  if (matrixRange.isImValues()) {
    aim = getMean(matrix, GetIm);
  }*/
  std::stringstream sstream1;
  std::stringstream sstream2;
  sstream1 << are;
  sstream2 << aim;
  output += message + "(" + sstream1.str() + "," + sstream2.str() + ") ";
}

math::Matrix* InfoCreatorHost::getMatrix(MatrixType matrixType) const {
  if (matrixType & InfoCreator::HOST) {
    if (matrixType & InfoCreator::EXPECTED) {
      return getExpectedMatrix();
    } else if (matrixType & InfoCreator::OUTPUT) {
      return getOutputMatrix();
    }
  }
  return NULL;
}

void InfoCreatorHost::getString(std::string& output, math::Matrix* matrix,
                                MatrixType matrixType) const {}

Complex InfoCreatorHost::getMean(math::Matrix* matrix,
                                 MatrixType matrixType) const {}

math::Matrix* InfoCreatorHost::createDiffMatrix(math::Matrix* matrix1,
                                                math::Matrix* matrix2,
                                                MatrixType matrixType) const {}

void InfoCreatorHost::destroyDiffMatrix(math::Matrix* diffMatrix) const {}

uintt InfoCreatorHost::getIndexOfLargestValue(math::Matrix* matrix) const {}

uintt InfoCreatorHost::getIndexOfSmallestValue(math::Matrix* matrix) const {}

/*
Complex InfoCreatorHost::getMean(math::Matrix* matrix, MatrixType matrixType)
const {
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

  Complex complex(are, aim);
  return complex;
}

InfoCreatorHost::InfoCreatorHost()
    : m_expected(NULL), m_output(NULL), m_diff(NULL), m_infoType(InfoType()) {}

InfoCreatorHost::InfoCreatorHost(math::Matrix* expected, math::Matrix* output,
                                 const InfoType& infoType)
    : m_expected(expected),
      m_output(output),
      m_diff(NULL),
      m_infoType(infoType) {}

InfoCreatorHost::~InfoCreatorHost() {}

void InfoCreatorHost::setExpected(math::Matrix* expected) {
  m_expected = expected;
}

void InfoCreatorHost::setOutput(math::Matrix* output) { m_output = output; }

void InfoCreatorHost::setInfoType(const InfoType& infoType) {
  m_infoType = infoType;
}

uintt Iter() {}

uintt getIndexOfLargestDiff(math::Matrix* m_diff,
                            floatt (*GetValue)(const math::Matrix*, uintt,
                                               uintt)) const {
  uintt length = m_diff->columns * m_diff->rows;
  if (length == 0) {
    return 0;
  }
  floatt largest = GetValue(m_diff, 0, 0);
  for (uintt fa = 0; fa < m_diff->columns; ++fa) {
    for (uintt fb = 0; fb < m_diff->rows; ++fb) {
      floatt value = GetValue(m_diff, fa, fb);
      if (largest < value) {
        largest = value;
      }
    }
  }
}

uintt getIndexOfSmallestDiff(math::Matrix* m_diff) const {}

uintt InfoCreatorHost::getIndexOfLargestDiff() const {
  if (diffMatrix != NULL) {
    return getIndexOfLargestDiff(m_diff);
  }
}

uintt InfoCreatorHost::getIndexOfSmallestDiff() const {
  if (diffMatrix != NULL) {
    return getIndexOfSmallestDiff(m_diff);
  }
}

floatt InfoCreatorHost::calculateMean(
    const matrixUtils::MatrixRange& matrixRange,
    floatt (*GetValue)(const math::Matrix*, uintt, uintt)) const {
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

void InfoCreatorHost::printInfo(std::string& info) {
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

bool InfoCreatorHost::isEquals() const { return m_isEqual; }*/
