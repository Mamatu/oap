#include "MathOperations.h"        
namespace math {

    Status IExpOperation::beforeExecution() {
        Status status = MatrixOperationOutputMatrix::beforeExecution();
        return status;
    }
}