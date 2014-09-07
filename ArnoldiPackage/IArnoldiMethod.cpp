/* 
 * File:   IArnoldiMethod.cpp
 * Author: mmatula
 * 
 * Created on August 25, 2014, 6:56 PM
 */

#include "IArnoldiMethod.h"

namespace math {

    IArnoldiMethod::IArnoldiMethod(MatrixModule* matrixModule,
            MatrixStructureUtils* matrixStructureUtils) :
    MatrixOperationOutputValues(matrixModule, matrixStructureUtils) {
    }

    IArnoldiMethod::~IArnoldiMethod() {
    }

    Status IArnoldiMethod::beforeExecution() {
        return STATUS_OK;
    }
}


