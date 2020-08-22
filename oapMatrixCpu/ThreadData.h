/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef OAP_INTERNAL_TYPES_H
#define	OAP_INTERNAL_TYPES_H
#include "MathOperations.h"
#include "ThreadsMapper.h"
#include "MatrixAPI.h"

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
            m_subcolumns = gColumns (matrix);
            m_subrows = gRows (matrix);
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

    void calculateRanges(uintt column, uintt row,
        uintt* bmap, uintt fa) {
        utils::mapper::ThreadsMap<uintt> tm;
        utils::mapper::getThreadsMap(tm, bmap, fa);
        calculateRanges(row, column, tm);
    }

    void calculateRanges(uintt column, uintt row,
        utils::mapper::ThreadsMap<uintt>& map) {
        begins[0] = column + map.beginColumn;
        begins[1] = row + map.beginRow;
        ends[0] = column + map.endColumn;
        ends[1] = row + map.endRow;
        offset = ends[0] - begins[0];
        assert(begins[0] < ends[0]);
        assert(begins[1] < ends[1]);
    }

    void calculateRanges(utils::mapper::ThreadsMap<uintt>& map) {
        calculateRanges(0, 0, map);
    }

    oap::utils::Thread thread;
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
