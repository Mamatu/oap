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

#include "Utils.h"
#include "MatchersImpl.h"

inline bool operator==(const math::Matrix& m1, const math::Matrix& m2) {
    if (m1.columns != m2.columns || m1.rows != m2.rows) {
        return false;
    }
    for (uintt fa = 0; fa < m1.columns; ++fa) {
        for (uintt fb = 0; fb < m1.rows; ++fb) {
            if (m1.reValues[fa + fb * m1.columns] != m2.reValues[fa + fb * m2.columns]) {
                return false;
            }
        }
    }
    return true;
}

MATCHER_P(MatrixIsEqual, matrix1, "") {
    return (*arg) == (*matrix1);
}

inline Matcher<math::Matrix*> MatrixValuesAreEqual(floatt value) {
    return MakeMatcher(new MatrixValuesAreEqualMatcher(value));
}

inline Matcher<floatt> IsEqualSum(floatt* buffer, size_t length, const std::string& extra = "") {
    return MakeMatcher(new BufferSumIsEqualMatcher<floatt>(buffer, length, extra));
}

inline Matcher<uintt> IsEqualSum(uintt* buffer, size_t length, const std::string& extra = "") {
    return MakeMatcher(new BufferSumIsEqualMatcher<uintt>(buffer, length, extra));
}

inline Matcher<int> IsEqualSum(int* buffer, size_t length, const std::string& extra = "") {
    return MakeMatcher(new BufferSumIsEqualMatcher<int>(buffer, length, extra));
}

inline Matcher<floatt> IsEqualSum(floatt sum, floatt* buffer, size_t length, const std::string& extra = "") {
    return MakeMatcher(new BufferSumIsEqualMatcherSum<floatt>(sum, buffer, length, extra));
}

inline Matcher<uintt> IsEqualSum(uintt sum, uintt* buffer, size_t length, const std::string& extra = "") {
    return MakeMatcher(new BufferSumIsEqualMatcherSum<uintt>(sum, buffer, length, extra));
}

inline Matcher<int> IsEqualSum(int sum, int* buffer, size_t length, const std::string& extra = "") {
    return MakeMatcher(new BufferSumIsEqualMatcherSum<int>(sum, buffer, length, extra));
}

MATCHER_P(MatrixValuesAreNotEqual, value, "") {
    return utils::areNotEqual(arg, value);
}

MATCHER_P(MatrixExIsEqual, buffer, "") {
    return utils::isEqual(arg, buffer);
}


#endif	/* MATRIXEQ_H */
