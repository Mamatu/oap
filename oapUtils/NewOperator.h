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
