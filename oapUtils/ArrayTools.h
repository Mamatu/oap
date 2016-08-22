/*
 * Copyright 2016 Marcin Matula
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




#ifndef ARRAYTOOLS_H
#define	ARRAYTOOLS_H

#include "DebugLogs.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>

#define GET_INDEX(type, vec, obj, index) \
type::iterator it = std::find(vec.begin(),vec.end(),obj);\
if(it == vec.end()) \
{ index = -1; } \
index = it - vec.begin();\

#define GET_ITERATOR(vec, obj, it) \
it = std::find(vec.begin(),vec.end(),obj);\


namespace ArrayTools {

    template<typename T, typename T1> bool set(T** dst, T1 dst_index,
            T1& dst_size, const T* src, T1 src_size) {
        if (dst == NULL) {
            return false;
        }
        if (src == NULL || src_size == 0) {
            *dst = NULL;
            dst_size = 0;
            return false;
        }
        if (dst_size - dst_index >= src_size) {
            memcpy((*dst) + dst_index, src, dst_size * sizeof (T));
            return true;
        }
        T* nts = new T[src_size + dst_index];
        memcpy(nts, dst, dst_index);
        memcpy(nts + dst_index, src, src_size);
        if (*dst != NULL) {
            delete[] (*dst);
        }
        (*dst) = nts;
        dst_size = src_size + dst_index;
        return true;
    }

    template<typename T, typename T1> bool set(T** dst, T1& dst_size,
            const T* src, T1 src_size) {
        return set(dst, T1(0), dst_size, src, src_size);
    }

    template<typename T, typename T1> bool set(T** dst, T1 dst_index,
            T1& dst_size, T& src) {
        const int src_size = 1;
        if (dst == NULL) {
            return false;
        }
        if (src == NULL || src_size == 0) {
            *dst = NULL;
            dst_size = 0;
            return false;
        }
        if (dst_size - dst_index >= src_size) {
            memcpy((*dst) + dst_index, src, dst_size * sizeof (T));
            return true;
        }
        T* nts = new T[src_size + dst_index];
        if (dst_index > 0) {
            memcpy(nts, dst, dst_index);
        }
        memcpy(nts + dst_index, src, src_size);
        if (*dst != NULL) {
            delete[] (*dst);
        }
        (*dst) = nts;
        dst_size = src_size + dst_index;
        return true;
    }

    template<typename T, typename T1> bool add(T** dst, T1& dst_size,
            const T* src, T1 src_size) {
        if (dst == NULL) {
            return false;
        }
        if (src_size == 0 || src == NULL) {
            return false;
        }
        T* nts = new T[dst_size + src_size];
        memcpy(nts, (*dst), dst_size * sizeof (T));
        memcpy(nts + dst_size, src, src_size);
        dst_size += src_size;
        if (*dst != NULL) {

            delete[] (*dst);
        }
        (*dst) = nts;
        return true;
    }

    template<typename T, typename T1> bool add(T** dst, T1& dst_size, T item) {
        if (dst == NULL) {
            return false;
        }
        T* nts = new T[dst_size + 1];
        memcpy(nts, (*dst), dst_size * sizeof (T));
        nts[dst_size] = item;
        dst_size += 1;
        if (*dst != NULL) {
            delete[] (*dst);
        }
        (*dst) = nts;
        return true;
    }

    template<typename T, typename T1> T1 indexOf(T* array, T1 size, T& item) {
        if (array == NULL || size == 0) {
            return -1;
        }
        for (int fa = 0; fa < size; fa++) {
            if (array[fa] == item) {
                return fa;
            }
        }
        return -1;
    }

    template<typename T, typename T1> bool remove(T** dst, T1& dst_size, T& item) {
        if (dst == NULL) {
            return false;
        }
        int index = indexOf(*dst, dst_size, item);
        if (index == -1) {
            return false;
        }
        T* nts = new T[dst_size - 1];
        memcpy(nts, (*dst), index);
        memcpy(nts + index, (*dst + index + 1), dst_size - index - 1);
        dst_size -= 1;
        if (*dst != NULL) {
            delete[] (*dst);
        }
        (*dst) = nts;
        return true;
    }

    template<typename T, typename T1> bool remove(T** dst, T1& dst_size,
            const T* items, T1 size) {
        if (size == 0 || items == NULL) {
            return false;
        }

        for (int fa = 0; fa < size; fa++) {
            remove(dst, dst_size, items[fa]);
        }
    }

    template<typename T, typename T1> void print(T* array, T1 size,
            const char* stext) {
        std::stringstream sstream;
        std::string text = "";
        for (int fa = 0; fa < size; fa++) {
            sstream << array[fa];
            text += sstream.str();
            sstream.str("");
            if (fa < size - 1) {
                text += ", ";
            }
        }
        printf("%s: %s \n", stext, text.c_str());
    }

    template<typename T, typename T1> T* create(T1 size, T value) {
        T* array = new T[size];
        for (int fa = 0; fa < size; fa++) {
            array[fa] = value;
        }
        return array;
    }

    template<typename T, typename T1> void clear(T* array, T1 size, T value) {
        for (int fa = 0; fa < size; fa++) {
            array[fa] = value;
        }
    }

    template<typename T, typename T1> int increment(T* array, T min, T max,
            T step, T1 index, T1 size) {
        int fa = index;
        int finished = 1;
        int finished1 = 0;
        do {
            finished = 1;
            array[fa]++;
            if (array[fa] == max) {
                array[fa] = min;
                fa++;
                finished = 0;
            }
            if (fa == size) {
                finished = 1;
                finished1 = 1;
            }
        } while (finished == 0);
        return fa;
    }

}

#define ADD_TO_VEC(vec, item) x.push_back(item)

#endif	/* ARRAYTOOLS_H */
