/* 
 * File:   matrixEq.h
 * Author: mmatula
 *
 * Created on December 23, 2014, 10:30 AM
 */

#ifndef MOCKUTILS_H
#define	MOCKUTILS_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Matrix.h"
#include "MatrixEx.h"

using ::testing::PrintToString;
using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;


namespace utils {

class Compare {
public:

    Compare();

    virtual ~Compare();

    virtual bool rule(const floatt& arg1, const floatt& arg2) = 0;

    bool compare(math::Matrix* matrix, floatt d);
};

bool isEqual(const MatrixEx& matrixEx, const uintt* buffer);

bool areEqual(math::Matrix* matrix, int d);

bool areNotEqual(math::Matrix* matrix, int d);

}

bool operator==(const math::Matrix& m1, const math::Matrix& m2);

MATCHER_P(MatrixIsEqual, matrix1, "") {
    return (*arg) == (*matrix1);
}

class MatrixValuesAreEqualMatcher : public MatcherInterface<math::Matrix*> {
    floatt m_value;
public:

    MatrixValuesAreEqualMatcher(floatt value);

    virtual bool MatchAndExplain(math::Matrix* matrix, MatchResultListener* listener) const;

    virtual void DescribeTo(::std::ostream* os) const;

    virtual void DescribeNegationTo(::std::ostream* os) const;
};

inline Matcher<math::Matrix*> MatrixValuesAreEqual(floatt value) {
  return MakeMatcher(new MatrixValuesAreEqualMatcher(value));
}

MATCHER_P(MatrixValuesAreNotEqual, value, "") {
    return utils::areNotEqual(arg, value);
}

MATCHER_P(MatrixExIsEqual, buffer, "") {
    return utils::isEqual(arg, buffer);
}


#endif	/* MATRIXEQ_H */

