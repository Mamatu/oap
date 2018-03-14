/*
 * Copyright 2016, 2017 Marcin Matula
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



#include <vector>

#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "Matrix.h"
#include "ThreadUtils.h"
#include "ThreadsMapper.h"
#include <math.h>

#define ReIsNotNull(m) m->reValues != NULL
#define ImIsNotNull(m) m->imValues != NULL

namespace math {

MathOperationsCpu::MathOperationsCpu()  {
    m_subrows[0] = 0;
    m_subrows[1] = MATH_UNDEFINED;
    m_subcolumns[0] = 0;
    m_subcolumns[1] = MATH_UNDEFINED;
    registerMathOperation(&m_additionOperation);
    registerMathOperation(&m_substracionOperation);
    registerMathOperation(&m_dotProductOperation);
    registerMathOperation(&m_tensorProductOperation);
    registerMathOperation(&m_diagonalizationOperation);
    registerMathOperation(&m_expOperation);
    registerMathOperation(&m_multiplicationConstOperation);
    registerMathOperation(&m_magnitudeOperation);
    registerMathOperation(&m_transposeOperation);
}

void MathOperationsCpu::registerMathOperation(IMathOperation* mathOperation) {
    operations.push_back(mathOperation);
    registerThreadsCountProperty(mathOperation);
}

void MathOperationsCpu::registerThreadsCountProperty(IMathOperation* mathOperation) {
    ThreadsCountProperty* threadsCountProperty =
        dynamic_cast<ThreadsCountProperty*> (mathOperation);
    if (threadsCountProperty) {
        properties.push_back(threadsCountProperty);
    }
}

MathOperationsCpu::~MathOperationsCpu() {
}

math::Status MathOperationsCpu::execute(math::TwoMatricesOperations& obj,
    math::Matrix* output, math::Matrix* arg1, math::Matrix* arg2) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputMatrix(output);
    obj.setMatrix1(arg1);
    obj.setMatrix2(arg2);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

math::Status MathOperationsCpu::execute(math::MatrixValueOperation& obj,
    math::Matrix* output, math::Matrix* arg1, floatt* value) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputMatrix(output);
    obj.setMatrix(arg1);
    obj.setReValue(value);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

math::Status MathOperationsCpu::execute(math::MatrixValueOperation& obj,
    math::Matrix* output, math::Matrix* arg1, floatt* revalue,
    floatt* imvalue) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputMatrix(output);
    obj.setMatrix(arg1);
    obj.setReValue(revalue);
    obj.setImValue(imvalue);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

math::Status MathOperationsCpu::execute(math::MatrixOperationOutputMatrix& obj,
    math::Matrix* output, math::Matrix* arg1) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputMatrix(output);
    obj.setMatrix(arg1);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

math::Status MathOperationsCpu::execute(math::MatrixOperationOutputValue& obj,
    floatt* output, math::Matrix* arg1) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputValue1(output);
    obj.setOutputValue2(NULL);
    obj.setMatrix(arg1);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

math::Status MathOperationsCpu::execute(math::MatrixOperationOutputValue& obj,
    floatt* output1, floatt* output2, math::Matrix* arg1) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputValue1(output1);
    obj.setOutputValue2(output2);
    obj.setMatrix(arg1);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

math::Status MathOperationsCpu::execute(math::MatrixOperationTwoOutputs& obj,
    math::Matrix* output1, math::Matrix* output2, math::Matrix* arg1) {
    obj.setSubRows(m_subrows);
    obj.setSubColumns(m_subcolumns);
    obj.setOutputMatrix1(output1);
    obj.setOutputMatrix2(output2);
    obj.setMatrix(arg1);
    math::Status status = obj.start();
    unsetSubRows();
    unsetSubColumns();
    if (status != 0) {
    }
    return status;
}

void MathOperationsCpu::setThreadsCount(int threadsCount) {
    //this->threadsCount = threadsCount;
    //this->additionOperation.setThreadsCount(this->threadsCount);
    //this->substracionOperation.setThreadsCount(this->threadsCount);
    //this->dotProductOperation.setThreadsCount(this->threadsCount);
    //this->tensorProductOperation.setThreadsCount(this->threadsCount);
    //this->diagonalizationOperation.setThreadsCount(this->threadsCount);
    //this->expOperation.setThreadsCount(this->threadsCount);
    for (unsigned int fa = 0; fa < properties.size(); fa++) {
        properties[fa]->setThreadsCount(threadsCount);
    }
}

void MathOperationsCpu::setSerieLimit(int serieLimit) {
    this->serieLimit = serieLimit;
}

void MathOperationsCpu::setSubRows(uintt subrows) {
    m_subrows[0] = 0;
    m_subrows[1] = subrows;
}

void MathOperationsCpu::setSubColumns(uintt subcolumns) {
    m_subcolumns[0] = 0;
    m_subcolumns[1] = subcolumns;
}

void MathOperationsCpu::setSubRows(uintt subrows[2]) {
    setSubRows(subrows[0], subrows[1]);
}

void MathOperationsCpu::setSubColumns(uintt subcolumns[2]) {
    setSubColumns(subcolumns[0], subcolumns[1]);
}

void MathOperationsCpu::setSubColumns(uintt begin, uintt end) {
    m_subcolumns[0] = begin;
    m_subcolumns[1] = end;
}

void MathOperationsCpu::setSubRows(uintt begin, uintt end) {
    m_subrows[0] = begin;
    m_subrows[1] = end;
}

void MathOperationsCpu::unsetSubRows() {
    this->m_subrows[0] = 0;
    this->m_subrows[1] = MATH_UNDEFINED;
    for (unsigned int fa = 0; fa < operations.size(); fa++) {
        operations[fa]->unsetSubRows();
    }
}

void MathOperationsCpu::unsetSubColumns() {
    this->m_subcolumns[0] = 0;
    this->m_subcolumns[1] = MATH_UNDEFINED;
    for (unsigned int fa = 0; fa < operations.size(); fa++) {
        operations[fa]->unsetSubColumns();
    }
}

void MathOperationsCpu::registerValueName(void* value, const std::string& name) {
#ifdef DEBUG_MATRIX_OPERATIONS
    valuesNames[value] = name;
#endif
}

math::Status MathOperationsCpu::add(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2) {
    math::Status status = execute(this->m_additionOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::substract(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2) {
    math::Status status = execute(this->m_substracionOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::dotProduct(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2) {
    math::Status status = execute(this->m_dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::dotProduct(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2, uintt offset) {
    m_dotProductOperation.setOffset(offset);
    math::Status status = execute(m_dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::dotProduct(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2, uintt boffset, uintt eoffset) {
    m_dotProductOperation.setOffset(boffset, eoffset);
    math::Status status = execute(m_dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::tensorProduct(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2) {
    math::Status status = execute(this->m_tensorProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::diagonalize(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2) {
    math::Status status = execute(this->m_diagonalizationOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::multiply(math::Matrix* output,
    math::Matrix* matrix1, floatt* value) {
    math::Status status = execute(this->m_multiplicationConstOperation, output, matrix1, value);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, value);
#endif
    return status;
}

math::Status MathOperationsCpu::multiply(math::Matrix* output,
    math::Matrix* matrix1, floatt* revalue, floatt* imvalue) {
    math::Status status = execute(this->m_multiplicationConstOperation,
        output, matrix1, revalue, imvalue);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, value);
#endif
    return status;
}

math::Status MathOperationsCpu::exp(math::Matrix* output, math::Matrix* matrix1) {
    math::Status status = execute(this->m_expOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1);
#endif
    return status;
}

math::Status MathOperationsCpu::multiply(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2) {
    math::Status status = execute(this->m_dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::multiply(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2, uintt offset) {
    m_dotProductOperation.setOffset(offset);
    math::Status status = execute(m_dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::multiply(math::Matrix* output,
    math::Matrix* matrix1, math::Matrix* matrix2,
    uintt boffset, uintt eoffset) {
    m_dotProductOperation.setOffset(boffset, eoffset);
    math::Status status = execute(m_dotProductOperation,
        output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
    return status;
}

math::Status MathOperationsCpu::magnitude(floatt* output, math::Matrix* matrix1) {
    math::Status status = execute(m_magnitudeOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1);
#endif
    return status;
}

math::Status MathOperationsCpu::transpose(math::Matrix* output, math::Matrix* matrix1) {
    math::Status status = execute(m_transposeOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1);
#endif
    return status;
}

math::Status MathOperationsCpu::transpose(math::Matrix* output, math::Matrix* matrix1,
    uintt subcolumns[2], uintt subrows[2]) {
    math::Status status = execute(m_transposeOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, output, matrix1);
#endif
    return status;
}

math::Status MathOperationsCpu::transpose(math::Matrix* matrix) {
#ifdef DEBUG_MATRIX_OPERATIONS            
    std::string matrixStr = "";
    oap::host::GetReMatrixStr(matrixStr, matrix);
#endif
    math::Status status = execute(m_transposeOperation, matrix, matrix);
#ifdef DEBUG_MATRIX_OPERATIONS
    printInfo(__FUNCTION__, matrix, matrixStr);
#endif
    return status;
}

math::Status MathOperationsCpu::det(floatt* output, math::Matrix* matrix) {
    return execute(m_determinantOperation, output, matrix);
}

math::Status MathOperationsCpu::det(floatt* output, floatt* output1, math::Matrix* matrix) {
    return execute(m_determinantOperation, output, output1, matrix);
}

math::Status MathOperationsCpu::qrDecomposition(math::Matrix* Q,
    math::Matrix* R, math::Matrix* matrix) {
    return execute(m_qrDecomposition, Q, R, matrix);
}



#define GET(x,y,index) x+index*y 

#define DEFAULT_CONSTRUCTOR(cname, bname) cname::cname():math::bname(){} cname::~cname(){}

#define DEFAULT_CONSTRUCTOR_WITH_ARGS(cname,bname,code) cname::cname():math::bname(){code} cname::~cname(){}

#define CHECK_PARAMS_PTR() if(this->output==NULL || matrix1 == NULL || matrix2 == NULL){return STATUS_INVALID_PARAMS;}
#define CHECK_PARAMS_PTR_1() if(this->output==NULL || matrix1 == NULL){return STATUS_INVALID_PARAMS;}
#define CHECK_PARAMS_PTR_3(a,b,c) if(a==NULL || b == NULL || c == NULL){return STATUS_INVALID_PARAMS;}

DEFAULT_CONSTRUCTOR(AdditionOperationCpu, IAdditionOperation);
DEFAULT_CONSTRUCTOR(SubstracionOperationCpu, ISubstracionOperation)
DEFAULT_CONSTRUCTOR(DotProductOperationCpu, IDotProductOperation);
DEFAULT_CONSTRUCTOR(TensorProductOperationCpu, ITensorProductOperation);
DEFAULT_CONSTRUCTOR(MagnitudeOperationCpu, IMagnitudeOperation);
DEFAULT_CONSTRUCTOR(DiagonalizationOperationCpu, IDiagonalizationOperation);
DEFAULT_CONSTRUCTOR(MultiplicationConstOperationCpu, IMultiplicationConstOperation);
DEFAULT_CONSTRUCTOR(ExpOperationCpu, IExpOperation);
//        DEFAULT_CONSTRUCTOR(QRDecomposition);
DEFAULT_CONSTRUCTOR(TransposeOperationCpu, ITransposeOperation);


}
