/*
 * Copyright 2016 - 2019 Marcin Matula
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



#include "MathOperations.h"


namespace math {

const char* getStr(math::Status status) {
    switch (status) {
        case math::STATUS_OK:
            return "STATUS_OK";
        case math::STATUS_INVALID_PARAMS:
            return "STATUS_INVALID_PARAMS";
        case math::STATUS_ERROR:
            return "STATUS_ERROR";
    };
}

const char* getErrorStr(math::Status status) {
    switch (status) {
        case math::STATUS_OK:
            return "";
        case math::STATUS_INVALID_PARAMS:
            return "STATUS_INVALID_PARAMS";
        case math::STATUS_ERROR:
            return "STATUS_ERROR";
    };
}
#define CLASS_BA(class_name, code, code1)\
        Status class_name::beforeExecution() {\
                code \
        }\
        Status class_name::afterExecution() {\
                code1 \
                return STATUS_OK;\
        }\


#define RESET_MATRIX(m) m->columns = m->realColumns; m->rows = m->realRows; 

CLASS_BA(TwoMatricesOperations,
if (this->m_output == NULL || this->m_matrix1 == NULL || this->m_matrix2 == NULL) {
    return STATUS_INVALID_PARAMS;
} else {
    return STATUS_OK;
},
RESET_MATRIX(m_output);
    RESET_MATRIX(m_matrix1);
    RESET_MATRIX(m_matrix2);
    );

CLASS_BA(MatrixValueOperation, if (this->m_output == NULL || this->m_matrix == NULL) {
    return STATUS_INVALID_PARAMS;
} else {
    return STATUS_OK;
},
RESET_MATRIX(m_output);
    RESET_MATRIX(m_matrix);
    );

CLASS_BA(MatrixOperationOutputValue, if (this->m_output1 == NULL || this->m_matrix == NULL) {
    return STATUS_INVALID_PARAMS;
} else {
    return STATUS_OK;
},
RESET_MATRIX(m_matrix);
    );

CLASS_BA(MatrixOperationOutputValues, if (this->m_reoutputs == NULL || this->m_matrix == NULL) {
    return STATUS_INVALID_PARAMS;
} else {
    return STATUS_OK;
},
RESET_MATRIX(m_matrix);
    );

CLASS_BA(MatrixOperationOutputMatrix, if (this->m_output == NULL || this->m_matrix == NULL) {
    return STATUS_INVALID_PARAMS;
} else {
    return STATUS_OK;
},
RESET_MATRIX(m_output);
    RESET_MATRIX(m_matrix);
    );
CLASS_BA(MatrixOperationTwoOutputs, if (this->m_output2 == NULL || this->m_output1 == NULL || this->m_matrix == NULL) {
    return STATUS_INVALID_PARAMS;
} else {
    return STATUS_OK;
},
RESET_MATRIX(m_output1);
    RESET_MATRIX(m_output2);
    RESET_MATRIX(m_matrix);
    );

#define CLASS_INITIALIZATOR(class_name, base_class_name)\
class_name::class_name(): \
        base_class_name() {\
}\
class_name::~class_name() {\
}\

#define CLASS_INITIALIZATOR_WITH_CODE(class_name, base_class_name, code)\
class_name::class_name(): \
        base_class_name() {\
        code \
}\
class_name::~class_name() {\
}\

CLASS_INITIALIZATOR(IAdditionOperation, TwoMatricesOperations);
CLASS_INITIALIZATOR(ISubstracionOperation, TwoMatricesOperations);
CLASS_INITIALIZATOR_WITH_CODE(IDotProductOperation, TwoMatricesOperations,
    m_offset[0] = 0; m_offset[1] = MATH_UNDEFINED;);
CLASS_INITIALIZATOR(IMultiplicationConstOperation, MatrixValueOperation);
CLASS_INITIALIZATOR(IExpOperation, MatrixOperationOutputMatrix);
CLASS_INITIALIZATOR(IDiagonalizationOperation, TwoMatricesOperations);
CLASS_INITIALIZATOR(ITensorProductOperation, TwoMatricesOperations);
CLASS_INITIALIZATOR(IMagnitudeOperation, MatrixOperationOutputValue);
CLASS_INITIALIZATOR(ITransposeOperation, MatrixOperationOutputMatrix);
CLASS_INITIALIZATOR(IDeterminantOperation, MatrixOperationOutputValue);
CLASS_INITIALIZATOR(IQRDecomposition, MatrixOperationTwoOutputs);

void IMathOperation::setSubRows(uintt subrows) {
    this->m_subrows[1] = subrows;
}

void IMathOperation::setSubColumns(uintt subcolumns) {
    this->m_subcolumns[1] = subcolumns;
}

void IMathOperation::setSubRows(uintt subrows[2]) {
    this->m_subrows[0] = subrows[0];
    this->m_subrows[1] = subrows[1];
}

void IMathOperation::setSubColumns(uintt subcolumns[2]) {
    this->m_subcolumns[0] = subcolumns[0];
    this->m_subcolumns[1] = subcolumns[1];
}

void IMathOperation::unsetSubRows() {
    this->m_subrows[0] = 0;
    this->m_subrows[1] = MATH_UNDEFINED;
}

void IMathOperation::unsetSubColumns() {
    this->m_subcolumns[0] = 0;
    this->m_subcolumns[1] = MATH_UNDEFINED;
}

bool IMathOperation::CopyIm(math::Matrix* dst, math::Matrix* src, IMathOperation *thiz) {
    if (thiz->m_subcolumns[1] == MATH_UNDEFINED &&
        thiz->m_subrows[1] == MATH_UNDEFINED) {
        //matrixCopier->copyImMatrixToImMatrix(dst, src);
    }
    bool b = true;//!matrixCopier->isError();
    return b;
}

bool IMathOperation::CopyRe(math::Matrix* dst, math::Matrix* src, IMathOperation *thiz) {
    if (thiz->m_subcolumns[1] == MATH_UNDEFINED &&
        thiz->m_subrows[1] == MATH_UNDEFINED) {
        //matrixCopier->copyReMatrixToReMatrix(dst, src);
    }
    bool b = true;// !matrixCopier->isError();
    return b;
}

bool IMathOperation::IsIm(math::Matrix* matrix) {
    //return matrixUtils->isImMatrix(matrix);
    return matrix->imValues != NULL;
}

bool IMathOperation::IsRe(math::Matrix* matrix) {
    //return matrixUtils->isReMatrix(matrix);
    return matrix->reValues != NULL;
}

IMathOperation::IMathOperation() {
    m_subrows[0] = 0;
    m_subrows[1] = MATH_UNDEFINED;
    m_subcolumns[0] = 0;
    m_subcolumns[1] = MATH_UNDEFINED;
}

IMathOperation::~IMathOperation() {
}

Status IMathOperation::start() {
    Status status = this->beforeExecution();
    if (status == STATUS_OK) {
        this->execute();
        status = this->afterExecution();
    }
    return status;
}

TwoMatricesOperations::TwoMatricesOperations() :
IMathOperation() {
    this->m_matrix1 = NULL;
    this->m_matrix2 = NULL;
    this->m_output = NULL;
}

TwoMatricesOperations::~TwoMatricesOperations() {
    // not implemented
}

MatrixValueOperation::MatrixValueOperation() :
IMathOperation() {
    this->m_matrix = NULL;
    this->m_output = NULL;
    this->m_revalue = NULL;
    this->m_imvalue = NULL;
}

MatrixValueOperation::~MatrixValueOperation() {
    // not implemented
}

MatrixOperationOutputMatrix::MatrixOperationOutputMatrix() :
IMathOperation() {
    this->m_matrix = NULL;
    this->m_output = NULL;
}

MatrixOperationOutputMatrix::~MatrixOperationOutputMatrix() {
    // not implemented
}

MatrixOperationOutputValue::MatrixOperationOutputValue() :
IMathOperation() {
    this->m_matrix = NULL;
    this->m_output1 = 0;
}

MatrixOperationOutputValue::~MatrixOperationOutputValue() {
    // not implemented
}

MatrixOperationOutputValues::MatrixOperationOutputValues() :
IMathOperation() {
    this->m_matrix = NULL;
    this->m_reoutputs = NULL;
    this->m_imoutputs = NULL;
    this->m_count = 0;
}

MatrixOperationOutputValues::~MatrixOperationOutputValues() {
    // not implemented
}

MatrixOperationTwoOutputs::MatrixOperationTwoOutputs() :
IMathOperation() {
    this->m_matrix = NULL;
    this->m_output1 = NULL;
    this->m_output2 = NULL;
}

MatrixOperationTwoOutputs::~MatrixOperationTwoOutputs() {
    // not implemented
}

void TwoMatricesOperations::setMatrix1(Matrix* matrix) {
    this->m_matrix1 = matrix;
}

void TwoMatricesOperations::setMatrix2(Matrix* matrix) {
    this->m_matrix2 = matrix;
}

void TwoMatricesOperations::setOutputMatrix(Matrix* matrix) {
    this->m_output = matrix;
}

void MatrixValueOperation::setMatrix(Matrix* matrix) {
    this->m_matrix = matrix;
}

void MatrixValueOperation::setReValue(floatt* value) {
    this->m_revalue = value;
}

void MatrixValueOperation::setImValue(floatt* value) {
    this->m_imvalue = value;
}

void MatrixValueOperation::setOutputMatrix(Matrix* matrix) {
    this->m_output = matrix;
}

void MatrixOperationOutputMatrix::setMatrix(Matrix* matrix) {
    this->m_matrix = matrix;
}

void MatrixOperationOutputMatrix::setOutputMatrix(Matrix* matrix) {
    this->m_output = matrix;
}

void MatrixOperationOutputValue::setMatrix(Matrix* matrix) {
    this->m_matrix = matrix;
}

void MatrixOperationOutputValue::setOutputValue1(floatt* value) {
    this->m_output1 = value;
}

void MatrixOperationOutputValue::setOutputValue2(floatt* value) {
    this->m_output2 = value;
}

void MatrixOperationOutputValues::setMatrix(Matrix* matrix) {
    this->m_matrix = matrix;
}

void MatrixOperationOutputValues::setReOutputValues(floatt* values, uintt count) {
    this->m_reoutputs = values;
    this->m_count = count;
}

void MatrixOperationOutputValues::setImOutputValues(floatt* values, uintt count) {
    this->m_imoutputs = values;
}

void MatrixOperationTwoOutputs::setMatrix(Matrix* matrix) {
    this->m_matrix = matrix;
}

void MatrixOperationTwoOutputs::setOutputMatrix1(Matrix* matrix) {
    this->m_output1 = matrix;
}

void MatrixOperationTwoOutputs::setOutputMatrix2(Matrix* matrix) {
    this->m_output2 = matrix;
}
}
