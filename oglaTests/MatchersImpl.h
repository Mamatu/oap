/* 
 * File:   MatchersImpl.h
 * Author: mmatula
 *
 * Created on March 22, 2015, 11:21 AM
 */

#ifndef MATCHERSIMPL_H
#define	MATCHERSIMPL_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Utils.h"

using ::testing::PrintToString;
using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

class MatrixValuesAreEqualMatcher : public MatcherInterface<math::Matrix*> {
    floatt m_value;
public:

    MatrixValuesAreEqualMatcher(floatt value) : m_value(value) {
    }

    virtual bool MatchAndExplain(math::Matrix* matrix, MatchResultListener* listener) const {
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

template<typename T> class BufferSumIsEqualMatcher : public MatcherInterface<T> {
protected:
    T* m_buffer;
    size_t m_length;
    T m_expectedSum;
    std::string m_extra;
    std::string m_stringRepresentation;
public:

    BufferSumIsEqualMatcher(T* buffer, size_t length, const std::string& extra = "") :
    m_buffer(buffer),
    m_length(length) {
        matrixUtils::PrintArray(m_stringRepresentation, m_buffer, m_length);
        m_expectedSum = utils::getSum(m_buffer, m_length);
        m_extra = extra;
    }

    virtual bool MatchAndExplain(T actualSum, MatchResultListener* listener) const {
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

template<typename T> class BufferSumIsEqualMatcherSum : public BufferSumIsEqualMatcher<T> {
public:

    BufferSumIsEqualMatcherSum(T expectedSum, T* buffer, size_t length, const std::string& extra = "") :
    BufferSumIsEqualMatcher<T>(buffer, length, extra) {
        BufferSumIsEqualMatcher<T>::m_expectedSum = expectedSum;
    }

    virtual bool MatchAndExplain(T actualSum, MatchResultListener* listener) const {
        std::string v;
        matrixUtils::PrintArray(v, BufferSumIsEqualMatcher<T>::m_buffer, BufferSumIsEqualMatcher<T>::m_length);
        (*listener) << v;
        return BufferSumIsEqualMatcher<T>::m_expectedSum == actualSum;
    }
};

#endif	/* MATCHERSIMPL_H */
