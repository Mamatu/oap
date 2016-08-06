#ifndef TESTPROCEDURES_H
#define TESTPROCEDURES_H

#include "KernelExecutor.h"

class CuTest {
    void* m_image;
    device::Kernel m_kernel;
    CUresult m_cuResult;
public:
    CuTest();
    bool test1();
    bool test2();
    CUresult getStatus() const;
};

#endif