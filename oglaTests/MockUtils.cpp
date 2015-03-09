/* 
 * File:   matrixEq.cpp
 * Author: mmatula
 * 
 * Created on December 23, 2014, 10:30 AM
 */

#include "MockUtils.h"
#include <string>
#include <vector>

namespace utils {

Compare::Compare() {
}

Compare::~Compare() {
}

bool Compare::compare(math::Matrix* matrix, floatt d) {
    if (NULL == matrix) {
        return false;
    }
    uintt length = matrix->rows * matrix->columns;
    for (uintt fa = 0; fa < length; ++fa) {
        if (!rule(matrix->reValues[fa], d)) {
            return false;
        }
    }
    return true;
}

bool isEqual(const MatrixEx& matrixEx, const uintt* buffer) {
    if (matrixEx.bcolumn == buffer[0]
        && matrixEx.ecolumn == buffer[1]
        && matrixEx.brow == buffer[2]
        && matrixEx.erow == buffer[3]
        && matrixEx.boffset == buffer[4]
        && matrixEx.eoffset == buffer[5]) {
        return true;
    }
    return false;
}

bool areEqual(math::Matrix* matrix, int d) {

    class CompareImpl : public Compare {
    public:

        bool rule(const floatt& arg1, const floatt& arg2) {
            return arg1 == arg2;
        }
    };
    CompareImpl compareImpl;
    return compareImpl.compare(matrix, d);
}

bool areNotEqual(math::Matrix* matrix, int d) {

    class CompareImpl : public Compare {
    public:

        bool rule(const floatt& arg1, const floatt& arg2) {
            return arg1 != arg2;
        }
    };
    CompareImpl compareImpl;
    return compareImpl.compare(matrix, d);
}

void PrintArray(std::string& output, floatt* array, size_t length) {
    std::vector<std::pair<uintt, floatt> > valuesVec;
    for (size_t fa = 0; fa < length; ++fa) {
        floatt value = array[fa];
        if (valuesVec.size() == 0 || valuesVec[valuesVec.size() - 1].second != value) {
            valuesVec.push_back(std::make_pair<uintt, floatt>(1, value));
        } else {
            valuesVec[valuesVec.size() - 1].first++;
        }
    }
    output = "[";
    for (size_t fa = 0; fa < valuesVec.size(); ++fa) {
        std::stringstream sstream;
        sstream << valuesVec[fa].second;
        if (valuesVec[fa].first > 0) {
            sstream << " times " << valuesVec[fa].first;
        }
        if (fa < valuesVec.size() - 1) {
            sstream << ", ";
        }
        output += sstream.str();
    }
    output += "]";
}

void PrintMatrix(std::string& output, math::Matrix* matrix) {
    std::stringstream sstream;
    sstream << "(" << matrix->columns << ", " << matrix->rows << ") ";
    output = sstream.str();
    size_t length = matrix->columns * matrix->rows;
    std::string output1;
    if (matrix->reValues != NULL) {
        PrintArray(output1, matrix->reValues, length);
        output += output1 + " ";
    }
    if (matrix->imValues != NULL) {
        PrintArray(output1, matrix->imValues, length);
        output += output1;
    }
}

}

bool operator==(const math::Matrix& m1, const math::Matrix& m2) {
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

MatrixValuesAreEqualMatcher::MatrixValuesAreEqualMatcher(floatt value) : m_value(value) {
}

bool MatrixValuesAreEqualMatcher::MatchAndExplain(math::Matrix* matrix, MatchResultListener* listener) const {
    std::string v;
    utils::PrintMatrix(v, matrix);
    (*listener) << v;
    return utils::areEqual(matrix, m_value);
}

void MatrixValuesAreEqualMatcher::DescribeTo(::std::ostream* os) const {
    *os << "Matrix values are equal " << m_value;
}

void MatrixValuesAreEqualMatcher::DescribeNegationTo(::std::ostream* os) const {
    *os << "Matrix values are not equal " << m_value;
}
