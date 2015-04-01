/* 
 * File:   Utils.h
 * Author: mmatula
 *
 * Created on March 22, 2015, 11:23 AM
 */

#ifndef UTILS_H
#define	UTILS_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Dim3.h"

#include "Matrix.h"
#include "MatrixEx.h"

namespace utils {

template <typename T> class OccurencesList : public std::vector<std::pair<uintt, T> > {
};

template<typename T> void PrepareOccurencesList(OccurencesList<T>& occurencesList,
    T* array, size_t length) {
    for (size_t fa = 0; fa < length; ++fa) {
        floatt value = array[fa];
        if (occurencesList.size() == 0 || occurencesList[occurencesList.size() - 1].second != value) {
            occurencesList.push_back(std::make_pair<uintt, floatt>(1, value));
        } else {
            occurencesList[occurencesList.size() - 1].first++;
        }
    }
}

template<typename T> void PrintArray(std::string& output, T* array, size_t length) {
    OccurencesList<T> valuesVec;
    PrepareOccurencesList(valuesVec, array, length);
    output = "[";
    std::stringstream sstream;
    for (size_t fa = 0; fa < valuesVec.size(); ++fa) {
        sstream << valuesVec[fa].second;
        if (valuesVec[fa].first > 0) {
            sstream << " times " << valuesVec[fa].first;
        }
        if (fa < valuesVec.size() - 1) {
            sstream << ", ";
        }
        output += sstream.str();
        sstream.str("");
    }
    sstream.str("");
    sstream << length;
    output += "] (length ";
    output += sstream.str();
    output += ")";
}

inline void PrintMatrix(std::string& output, math::Matrix* matrix) {
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

class Compare {
public:

    Compare() {
    }

    virtual ~Compare() {
    }


    virtual bool rule(const floatt& arg1, const floatt& arg2) = 0;

    inline bool compare(math::Matrix* matrix, floatt d) {
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
};

inline bool isEqual(const MatrixEx& matrixEx, const uintt* buffer) {
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

inline bool areEqual(math::Matrix* matrix, int d) {

    class CompareImpl : public Compare {
    public:

        bool rule(const floatt& arg1, const floatt& arg2) {
            return arg1 == arg2;
        }
    };
    CompareImpl compareImpl;
    return compareImpl.compare(matrix, d);
}

inline bool areNotEqual(math::Matrix* matrix, int d) {

    class CompareImpl : public Compare {
    public:

        bool rule(const floatt& arg1, const floatt& arg2) {
            return arg1 != arg2;
        }
    };
    CompareImpl compareImpl;
    return compareImpl.compare(matrix, d);
}

typedef std::pair<size_t, size_t> Range;

template<typename T> class Ranges : public std::vector<std::pair<T, Range> > {
};

template<typename T> bool isMatched(T* array, size_t length,
    const Ranges<T>& matched, const Ranges<T>& notmatched) {
}

//template<typename T> typedef typename std::vector<std::pair<T, Range<T> > > Ranges<T>;

//template<typename T> bool isMatched(T* array, size_t length) {

//}

template<typename T> T getSum(T* buffer, size_t length) {
    T output = 0;
    for (uintt fa = 0; fa < length; ++fa) {
        output += buffer[fa];
    }
    return output;
}

#ifdef CUDATEST

inline std::string gridDimToStr(const Dim3& dim3) {
    std::ostringstream s;
    s << "gridDim = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
    return s.str();
}

inline std::string blockDimToStr(const Dim3& dim3) {
    std::ostringstream s;
    s << "blockDim = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
    return s.str();
}

inline std::string blockIdxToStr(const Dim3& dim3) {
    std::ostringstream s;
    s << "blockIdx = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
    return s.str();
}

inline std::string threadIdxToStr(const Dim3& dim3) {
    std::ostringstream s;
    s << "threadIdx = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
    return s.str();
}

inline std::string cudaDimsToStr() {
    std::string output = " ";
    output += threadIdxToStr(threadIdx) + " ";
    output += blockIdxToStr(blockIdx) + " ";
    output += blockDimToStr(blockDim) + " ";
    output += gridDimToStr(gridDim) + " ";
    return output;
}
#endif

}

#endif	/* UTILS_H */
