#ifndef OGLA_INTERNAL_TYPES_H
#define	OGLA_INTERNAL_TYPES_H
#include "MathOperations.h"
#include "ThreadsMapper.h"

#define MAX_OUTPUTS_NUMBER 2
#define MAX_PARAMS_NUMBER 2
#define MAX_VALUES_NUMBER 2
#define MAX_BEGINS_NUMBER 4
#define MAX_ENDS_NUMBER 4

template<typename T> class ThreadData {
public:

    class SubMatrix {
    public:

        SubMatrix() {
            m_matrix = NULL;
            m_subcolumns = 0;
            m_subrows = 0;
        }

        SubMatrix& operator=(math::Matrix* matrix) {
            m_matrix = matrix;
            m_subcolumns = matrix->columns;
            m_subrows = matrix->rows;
            return *this;
        }
        
        math::Matrix* operator->() const {
            return m_matrix;
        }

        math::Matrix* m_matrix;
        uintt m_subrows;
        uintt m_subcolumns;
    };

    ThreadData(void* userDataPtr = NULL) : m_userDataPtr(userDataPtr) {
        memset(valuesPtr, 0, MAX_VALUES_NUMBER * sizeof (floatt));
        memset(values, 0, MAX_VALUES_NUMBER * sizeof (floatt));
        memset(begins, 0, MAX_BEGINS_NUMBER * sizeof (int));
        memset(ends, 0, MAX_ENDS_NUMBER * sizeof (int));
    }

    void calculateRanges(uintt* bmap, uintt fa) {
        utils::mapper::ThreadsMap<uintt> tm;
        utils::mapper::getThreadsMap(tm, bmap, fa);
        calculateRanges(0, 0, tm);
    }

    void calculateRanges(uintt columns[2], uintt rows[2],
            uintt* bmap, uintt fa) {
        utils::mapper::ThreadsMap<uintt> tm;
        utils::mapper::getThreadsMap(tm, bmap, fa);
        calculateRanges(columns[0], rows[0], tm);
    }

    void calculateRanges(uintt beginColumn, uintt beginRow,
            uintt* bmap, uintt fa) {
        utils::mapper::ThreadsMap<uintt> tm;
        utils::mapper::getThreadsMap(tm, bmap, fa);
        calculateRanges(beginRow, beginColumn, tm);
    }

    void calculateRanges(uintt beginColumn, uintt beginRow,
            utils::mapper::ThreadsMap<uintt>& map) {
        begins[0] = beginColumn + map.beginColumn;
        begins[1] = beginRow + map.beginRow;
        ends[0] = beginColumn + map.endColumn;
        ends[1] = beginRow + map.endRow;
        offset = ends[0] - begins[0];
    }

    void calculateRanges(utils::mapper::ThreadsMap<uintt>& map) {
        calculateRanges(0, 0, map);
    }

    utils::Thread thread;
    void* m_userDataPtr;
    int index;
    SubMatrix outputs[MAX_OUTPUTS_NUMBER];
    SubMatrix params[MAX_PARAMS_NUMBER];
    floatt* valuesPtr[MAX_VALUES_NUMBER];
    Complex* complexPtr[MAX_VALUES_NUMBER];
    floatt values[MAX_VALUES_NUMBER];
    intt begins[MAX_BEGINS_NUMBER];
    intt ends[MAX_ENDS_NUMBER];
    intt offset;
    T* thiz;
};

#endif	/* INTERNALTYPES_H */

