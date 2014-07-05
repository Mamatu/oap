/* 
 * File:   LHandle.cpp
 * Author: mmatula
 * 
 * Created on April 17, 2014, 9:16 PM
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

