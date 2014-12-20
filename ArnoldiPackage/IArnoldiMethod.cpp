/* 
 * File:   IArnoldiMethod.cpp
 * Author: mmatula
 * 
 * Created on August 25, 2014, 6:56 PM
 */

#include "IArnoldiMethod.h"

namespace math {

    IArnoldiMethod::IArnoldiMethod(MatrixModule* matrixModule) :
    MatrixOperationOutputValues(matrixModule) {
    }

    IArnoldiMethod::~IArnoldiMethod() {
    }

    Status IArnoldiMethod::beforeExecution() {
        return STATUS_OK;
    }
}


