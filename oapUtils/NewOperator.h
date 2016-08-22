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



#ifndef NEWOPERATOR_H
#define NEWOPERATOR_H

#include <new>
#include <stdio.h>

/*
void* operator new(std::size_t sz, const std::nothrow_t&)
{
    void* ptr = ::operator new(sz);
    printf("new operator %p %u \n", ptr, sz);
    return ptr;
}

void* operator new[](std::size_t sz, const std::nothrow_t&)
{
    void* ptr = ::operator new(sz);
    printf("new operator %p %u \n", sz, ptr);
    return ptr;
}


void* operator new(std::size_t sz)
{
    void* ptr = ::operator new(sz);
    printf("new operator %p %u \n", ptr, sz);
    return ptr;
}

void* operator new[](std::size_t sz)
{
    void* ptr = ::operator new(sz);
    printf("new operator %p %u \n", sz, ptr);
    return ptr;
}


void operator delete(void* data)
{
    ::operator delete(data);
    printf("delete operator %p \n", data);
}

void operator delete[](void* data)
{
    ::operator delete[](data);
    printf("delete operator %p \n", data);
}
*/
#endif  // NEWOPERATOR_H
