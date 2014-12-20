/* 
 * File:   IArnoldiMethod.h
 * Author: mmatula
 *
 * Created on August 25, 2014, 6:56 PM
 */

#ifndef IARNOLDIMETHOD_H
#define	IARNOLDIMETHOD_H

#include "MathOperations.h"

namespace math {

    class IArnoldiMethod : public MatrixOperationOutputValues {
    protected:
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IArnoldiMethod(MatrixModule* matrixModule);
        virtual ~IArnoldiMethod();
        virtual void setHSize(uintt k) = 0;
        virtual void setRho(floatt rho) = 0;
    };
}

#endif	/* IARNOLDIMETHOD_H */

