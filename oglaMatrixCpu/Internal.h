#ifndef OGLA_INTERNAL_TYPES_H
#define	OGLA_INTERNAL_TYPES_H
#include "MathOperations.h"
#include "ThreadsMapper.h"

#define MAX_OUTPUTS_NUMBER 2
#define MAX_PARAMS_NUMBER 2
#define MAX_VALUES_NUMBER 2
#define MAX_BEGINS_NUMBER 4
#define MAX_ENDS_NUMBER 4

#define IS_DEFINED(sub) (sub[0] != -1 && sub[1] != -1)

template<typename T> class ThreadData {
public:

    ThreadData(void* userDataPtr = NULL) : m_userDataPtr(userDataPtr) {
        memset(valuesPtr, 0, MAX_VALUES_NUMBER * sizeof (floatt));
        memset(values, 0, MAX_VALUES_NUMBER * sizeof (floatt));
        memset(begins, 0, MAX_BEGINS_NUMBER * sizeof (int));
        memset(ends, 0, MAX_ENDS_NUMBER * sizeof (int));
    }

    void calculateRanges(const math::Matrix* output, uintt* bmap,
            uintt fa) {
        utils::mapper::ThreadsMap<uintt> tm;
        utils::mapper::getThreadsMap(tm, bmap, fa);
        calculateRanges(output, tm);
    }

    void calculateRanges(const math::Matrix* output,
            utils::mapper::ThreadsMap<uintt>& map) {
        begins[0] = map.beginColumn;
        begins[1] = map.beginRow;
        ends[0] = map.endColumn;
        ends[1] = map.endRow;
        offset = ends[0] - begins[0];
    }

    utils::Thread thread;
    void* m_userDataPtr;
    int index;
    math::Matrix* outputs[MAX_OUTPUTS_NUMBER];
    math::Matrix* params[MAX_PARAMS_NUMBER];
    floatt* valuesPtr[MAX_VALUES_NUMBER];
    Complex* complexPtr[MAX_VALUES_NUMBER];
    floatt values[MAX_VALUES_NUMBER];
    intt begins[MAX_BEGINS_NUMBER];
    intt ends[MAX_ENDS_NUMBER];
    intt offset;
    T* thiz;
};

#endif	/* INTERNALTYPES_H */

