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




#ifndef LHANDLE_H
#define	LHANDLE_H


#define BYTES_COUNT 8

/**
 * 
 */
class LHandle {
    int bytesCount;
    char byteRepresentation[BYTES_COUNT];
    void clear();
public:

    /**
     * Return pointer stored in this object.
     * Warning! If length of pointer which is stored in this class
     * is different than sizeof(void*) this method return NULL/
     *  @return Pointer which is stored in this object or NULL.
     */
    void* getPtr() const;

    LHandle();

    LHandle(void* ptr) ;

    LHandle(const LHandle& lHandle);

    LHandle& operator=(const LHandle& lhandle);

    bool operator==(const LHandle& lhandle);

    bool lessThan(const LHandle& lhandle) const;
};

bool operator<(const LHandle& handle1, const LHandle& handle2);
#endif	/* LHANDLE_H */
