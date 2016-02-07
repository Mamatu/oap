/*
 * File:   MatchersImpl.h
 * Author: mmatula
 *
 * Created on March 22, 2015, 11:21 AM
 */

#ifndef MATCHERSIMPL_H
#define MATCHERSIMPL_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "MatrixTestAPI.h"
#include "InfoCreator.h"
#include "Utils.h"

using ::testing::PrintToString;
using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

class MatrixValuesAreEqualMatcher : public MatcherInterface<math::Matrix*> {
  floatt m_value;

 public:
  MatrixValuesAreEqualMatcher(floatt value) : m_value(value) {}

  virtual bool MatchAndExplain(math::Matrix* matrix,
                               MatchResultListener* listener) const {
    std::string v;
    matrixUtils::PrintMatrix(v, matrix);
    (*listener) << v;
    return utils::areEqual(matrix, m_value);
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "Matrix values are equal " << m_value;
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "Matrix values are not equal " << m_value;
  }
};

class MatrixIsEqualMatcher : public MatcherInterface<math::Matrix*> {
  math::Matrix* m_matrix;
  InfoType m_infoType;

 public:
  MatrixIsEqualMatcher(math::Matrix* matrix, const InfoType& infoType)
      : m_matrix(matrix), m_infoType(infoType) {}

  virtual bool MatchAndExplain(math::Matrix* matrix,
                               MatchResultListener* listener) const {
    InfoCreator infoCreator(m_matrix, matrix, m_infoType);
    std::string msg;
    infoCreator.printInfo(msg);
    (*listener) << msg;
    return infoCreator.isEquals();
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "Matrices are equal.";
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "Matrices are not equal.";
  }
};

class MatrixIsDiagonalMatcher : public MatcherInterface<math::Matrix*> {
  floatt m_value;

 public:
  MatrixIsDiagonalMatcher(floatt value) : m_value(value) {}

  virtual bool MatchAndExplain(math::Matrix* matrix,
                               MatchResultListener* listener) const {
    math::Matrix* diffmatrix = NULL;
    std::string matrixStr;
    bool isequal = utils::IsDiagonalMatrix((*matrix), m_value, &diffmatrix);
    matrixUtils::PrintMatrix(matrixStr, diffmatrix);
    if (!isequal) {
      (*listener) << "Diff is = " << matrixStr;
    }
    host::DeleteMatrix(diffmatrix);
    return isequal;
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "Matrix is diagonal.";
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "Matrix is not diagonal.";
  }
};

class MatrixIsIdentityMatcher : public MatrixIsDiagonalMatcher {
 public:
  MatrixIsIdentityMatcher() : MatrixIsDiagonalMatcher(1.f) {}

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "Matrix is identity.";
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "Matrix is not identity.";
  }
};

class MatrixTestAPIMatcher : public MatcherInterface<math::Matrix*> {
  bool (*m_checker)(const math::Matrix* matrix);
  uintt (*m_getter)(const math::Matrix* matrix);

 public:
  MatrixTestAPIMatcher(bool (*checker)(const math::Matrix* matrix),
                       uintt (*getter)(const math::Matrix* matrix))
      : m_getter(getter), m_checker(checker) {}

  virtual bool MatchAndExplain(math::Matrix* matrix,
                               MatchResultListener* listener) const {
    bool is = (*m_checker)(matrix);
    if (!is) {
      uintt count = (*m_getter)(matrix);
      (*listener) << "were set " << count << " elements";
    }
    return is;
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "were set all elements.";
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "were not set all elemets.";
  }
};

class SetAllRe : public MatrixTestAPIMatcher {
 public:
  SetAllRe()
      : MatrixTestAPIMatcher(test::wasSetAllRe, test::getSetValuesCountRe) {}
};

class SetAllIm : public MatrixTestAPIMatcher {
 public:
  SetAllIm()
      : MatrixTestAPIMatcher(test::wasSetAllIm, test::getSetValuesCountIm) {}
};

class GetAllRe : public MatrixTestAPIMatcher {
 public:
  GetAllRe()
      : MatrixTestAPIMatcher(test::wasGetAllRe, test::getGetValuesCountRe) {}
};

class GetAllIm : public MatrixTestAPIMatcher {
 public:
  GetAllIm()
      : MatrixTestAPIMatcher(test::wasGetAllIm, test::getGetValuesCountIm) {}
};

template <typename T>
class BufferSumIsEqualMatcher : public MatcherInterface<T> {
 protected:
  T* m_buffer;
  size_t m_length;
  T m_expectedSum;
  std::string m_extra;
  std::string m_stringRepresentation;

 public:
  BufferSumIsEqualMatcher(T* buffer, size_t length,
                          const std::string& extra = "")
      : m_buffer(buffer), m_length(length) {
    matrixUtils::PrintArray<T>(m_stringRepresentation, m_buffer, m_length,
                               static_cast<T>(0));
    m_expectedSum = utils::getSum(m_buffer, m_length);
    m_extra = extra;
  }

  virtual bool MatchAndExplain(T actualSum,
                               MatchResultListener* listener) const {
    (*listener) << m_stringRepresentation;
    return m_expectedSum == actualSum;
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "Sum of array is equal " << m_expectedSum << m_extra;
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "Sum of array is not equal " << m_expectedSum << m_extra;
  }
};

template <typename T>
class BufferSumIsEqualMatcherSum : public BufferSumIsEqualMatcher<T> {
 public:
  BufferSumIsEqualMatcherSum(T expectedSum, T* buffer, size_t length,
                             const std::string& extra = "")
      : BufferSumIsEqualMatcher<T>(buffer, length, extra) {
    BufferSumIsEqualMatcher<T>::m_expectedSum = expectedSum;
  }

  virtual bool MatchAndExplain(T actualSum,
                               MatchResultListener* listener) const {
    std::string v;
    matrixUtils::PrintArray(v, BufferSumIsEqualMatcher<T>::m_buffer,
                            BufferSumIsEqualMatcher<T>::m_length,
                            static_cast<T>(0));
    (*listener) << v;
    return BufferSumIsEqualMatcher<T>::m_expectedSum == actualSum;
  }
};

#endif /* MATCHERSIMPL_H */
