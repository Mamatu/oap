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




#include "LHandle.h"
#include <string.h>

void LHandle::clear() {
    memset(this->byteRepresentation, 0, this->bytesCount);
}

void* LHandle::getPtr() const {
    void* ptr = NULL;
    if (sizeof (void*) == this->bytesCount) {
        memcpy(ptr, this->byteRepresentation, this->bytesCount);
    } else {
    }
    return ptr;
}

LHandle::LHandle() : bytesCount(BYTES_COUNT) {
    clear();
}

LHandle::LHandle(void* ptr) : bytesCount(sizeof (ptr)) {
    clear();
    memcpy(byteRepresentation, ptr, sizeof (void*));
}

LHandle::LHandle(const LHandle& lHandle) : bytesCount(sizeof (lHandle.bytesCount)) {
    clear();
    memcpy(this->byteRepresentation, lHandle.byteRepresentation, lHandle.bytesCount);
}

LHandle& LHandle::operator=(const LHandle& lhandle) {
    clear();
    this->bytesCount = bytesCount;
    memcpy(this->byteRepresentation, lhandle.byteRepresentation, lhandle.bytesCount);
}

bool LHandle::operator==(const LHandle& lhandle) {
    if (this->bytesCount != lhandle.bytesCount) {
        return false;
    }
    return this == &lhandle || strncmp(this->byteRepresentation, lhandle.byteRepresentation, lhandle.bytesCount);
}

bool LHandle::lessThan(const LHandle& lhandle) const {
    if (lhandle.bytesCount != this->bytesCount) {
        return this->bytesCount < lhandle.bytesCount;
    }
    for (int fa = 0; fa<this->bytesCount; fa++) {
        if (this->byteRepresentation[fa] != lhandle.byteRepresentation[fa]) {
            return this->byteRepresentation[fa] < lhandle.byteRepresentation[fa];
        }
    }
    return false;
}

bool operator<(const LHandle& handle1, const LHandle& handle2) {
    return handle1.lessThan(handle2);
}
