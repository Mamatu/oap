/* 
 * File:   oglaCudaUtils.h
 * Author: mmatula
 *
 * Created on March 3, 2015, 11:15 PM
 */

#ifndef OGLACUDASTUB_H
#define	OGLACUDASTUB_H

#include "gtest/gtest.h"
#include "Math.h"
#include "ThreadsMapper.h"

class OglaCudaStub : public testing::Test {
public:
    OglaCudaStub();
    OglaCudaStub(const OglaCudaStub& orig);
    virtual ~OglaCudaStub();

    class Dim3 {
    public:

        Dim3() {
            x = 0;
            y = 0;
            z = 1;
        }

        Dim3(size_t tuple[2]) {
            x = tuple[0];
            y = tuple[1];
            z = 1;
        }

        Dim3(uintt tuple[2]) {
            x = tuple[0];
            y = tuple[1];
            z = 1;
        }

        Dim3(size_t x, size_t y) {
            this->x = x;
            this->y = y;
            z = 1;
        }

        size_t x;
        size_t y;
        size_t z;
    };

    class KernelStub {
    protected:
        Dim3 threadIdx;
        Dim3 blockIdx;
        Dim3 blockDim;
        Dim3 gridDim;
    public:

        virtual ~KernelStub() {
        }

        void setDims(const Dim3& blockDim, const Dim3& gridDim);

        void calculateDims(uintt columns, uintt rows);

    protected:
        virtual void execute() = 0;

        friend class OglaCudaStub;
    };

    void executeKernelSync(KernelStub* cudaStub);
};

#endif	/* OGLACUDAUTILS_H */

