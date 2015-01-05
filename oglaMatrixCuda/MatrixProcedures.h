#ifndef OGLA_CUMATRIXPROCEDURES_H
#define	OGLA_CUMATRIXPROCEDURES_H

#include "Matrix.h"
#include "MatrixEx.h"

void CuMatrix_dotProduct(math::Matrix* ouput,
    math::Matrix* params0, math::Matrix* params1);

void CuMatrix_dotProductEx(math::Matrix* ouput,
    math::Matrix* params0, math::Matrix* params1,
    MatrixEx* matrixEx);

void CuMatrix_transposeMatrixEx(math::Matrix* output,
    math::Matrix* params0, MatrixEx* matrixEx);

void CuMatrix_transposeMatrix(math::Matrix* output,
    math::Matrix* params0);

void CuMatrix_substractMatrix(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1);

void CuMatrix_addMatrix(math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1);

void CuMatrix_setVector(math::Matrix* output, uintt index,
    math::Matrix* params0, uintt length);

void CuMatrix_getVector(math::Matrix* vector, uintt rows,
    math::Matrix* matrix, uintt column);

void CuMatrix_magnitude(floatt& output, math::Matrix* params0);

void CuMatrix_multiplyConstantMatrix(math::Matrix* v,
    math::Matrix* f, floatt re);

void CuMatrix_multiplyConstantMatrix(math::Matrix* v,
    math::Matrix* f, floatt re, floatt im);

void CuMatrix_setDiagonalMatrix(math::Matrix* matrix, floatt* re, floatt* im);

void CuMatrix_setIdentity(math::Matrix* matrix);

void CuMatrix_setZeroMatrix(math::Matrix* matrix);

void CuMatrix_QR(math::Matrix* Q,
    math::Matrix* R, math::Matrix* H,
    math::Matrix* R1, math::Matrix* Q1,
    math::Matrix* G, math::Matrix * GT);


#endif	/* MATRIXPROCEDURES_H */

