#include "InfoCreator.h"
#include "MatrixUtils.h"

InfoCreator::InfoCreator() {}

InfoCreator::~InfoCreator() {}

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

void InfoCreator::getInfo(std::string& outputStr,
                          const InfoType& infoType) const {
  math::Matrix* output = getOutputMatrix();
  math::Matrix* expected = getExpectedMatrix();
  math::Matrix* diffmatrix = createDiffMatrix(expected, output);
  bool isequal = diffmatrix == NULL;
  if (!isequal) {
    if (infoType.getInfo() & InfoType::MEAN) {
      printMean(outputStr);
      outputStr += "\n";
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
      printMatrix(outputStr, "Output = ", m_output);
      printMatrix(outputStr, "Expected = ", m_expected);
      printMatrix(outputStr, "Diff = ", diffmatrix);
      outputStr += "\n";
    }
  }
  destroyDiffMatrix(diffmatrix);
}

void InfoCreator::printMatrix(std::string& output, const std::string& message,
                              math::Matrix* matrix) const {
  std::string matrixStr;
  getString(matrixStr, matrix);
  output += message + matrixStr;
}

bool InfoCreator::printMean(std::string& output) const {
  math::Matrix* expectedMatrix = getExpectedMatrix();
  math::Matrix* outputMatrix = getOutputMatrix();
  if (expectedMatrix == NULL || outputMatrix == NULL) {
    return false;
  }
  math::Matrix* diff = createDiffMatrix(expectedMatrix, outputMatrix);
  printMean(output, "Expected mean = ", expectedMatrix);
  printMean(output, "Output mean = ", outputMatrix);
  printMean(output, "Diff mean = ", diff);
  destroyDiffMatrix(diff);
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

bool InfoCreator::isEqual() const {
  math::Matrix* output = getOutputMatrix();
  math::Matrix* expected = getExpectedMatrix();
  return compare(output, expected);
}

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
