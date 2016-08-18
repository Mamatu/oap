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


