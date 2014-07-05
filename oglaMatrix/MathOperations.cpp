/* 
 * File:   MathOperations.cpp
 * Author: mmatula
 * 
 * Created on March 22, 2014, 12:15 PM
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
#define CLASS_BA(class_name, code)\
        Status class_name::beforeExecution() {\
                code \
        }\
        Status class_name::afterExecution() {\
                return STATUS_OK;\
        }\
        
    CLASS_BA(TwoMatricesOperations,
    if (this->m_output == NULL || this->m_matrix1 == NULL || this->m_matrix2 == NULL) {
        return STATUS_INVALID_PARAMS;
    } else {
        return STATUS_OK;
    });

    CLASS_BA(MatrixValueOperation, if (this->m_output == NULL || this->m_matrix == NULL) {
        return STATUS_INVALID_PARAMS;
    } else {
        return STATUS_OK;
    });
    
    CLASS_BA(MatrixOperationOutputValue, if (this->m_output1 == NULL || this->m_matrix == NULL) {
        return STATUS_INVALID_PARAMS;
    } else {
        return STATUS_OK;
    });

    CLASS_BA(MatrixOperationOutputValues, if (this->m_reoutputs == NULL || this->m_matrix == NULL) {
        return STATUS_INVALID_PARAMS;
    } else {
        return STATUS_OK;
    });

    CLASS_BA(MatrixOperationOutputMatrix, if (this->m_output == NULL || this->m_matrix == NULL) {
        return STATUS_INVALID_PARAMS;
    } else {
        return STATUS_OK;
    });
    CLASS_BA(MatrixOperationTwoOutputs, if (this->m_output2 == NULL || this->m_output1 == NULL || this->m_matrix == NULL) {
        return STATUS_INVALID_PARAMS;
    } else {
        return STATUS_OK;
    });

#define CLASS_INITIALIZATOR(class_name, base_class_name)\
        class_name::class_name(MatrixModule* _matrixModule,\
            MatrixStructureUtils* _matrixStructureUtils): \
                base_class_name(_matrixModule, _matrixStructureUtils) {\
        }\
        class_name::~class_name() {\
        }\


    CLASS_INITIALIZATOR(IAdditionOperation, TwoMatricesOperations);
    CLASS_INITIALIZATOR(ISubstracionOperation, TwoMatricesOperations);
    CLASS_INITIALIZATOR(IDotProductOperation, TwoMatricesOperations);
    CLASS_INITIALIZATOR(IMultiplicationConstOperation, MatrixValueOperation);
    CLASS_INITIALIZATOR(IExpOperation, MatrixOperationOutputMatrix);
    CLASS_INITIALIZATOR(IDiagonalizationOperation, TwoMatricesOperations);
    CLASS_INITIALIZATOR(ITensorProductOperation, TwoMatricesOperations);
    CLASS_INITIALIZATOR(IMagnitudeOperation, MatrixOperationOutputValue);
    CLASS_INITIALIZATOR(ITransposeOperation, MatrixOperationOutputMatrix);
    CLASS_INITIALIZATOR(IDeterminantOperation, MatrixOperationOutputValue);
    CLASS_INITIALIZATOR(IIraMethod, MatrixOperationOutputValues);
    CLASS_INITIALIZATOR(IQRDecomposition, MatrixOperationTwoOutputs);

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
        this->m_subrows[1] = 0;
    }

    void IMathOperation::unsetSubColumns() {
        this->m_subcolumns[0] = 0;
        this->m_subcolumns[1] = 0;
    }

    bool IMathOperation::CopyIm(math::Matrix* dst, math::Matrix* src, MatrixCopier* matrixCopier, IMathOperation *thiz) {
        if (thiz->m_subcolumns[0] == -1 && thiz->m_subrows[0] == -1) {
            matrixCopier->copyImMatrixToImMatrix(dst, src);
        }
        bool b = !matrixCopier->isError();
        return b;
    }

    bool IMathOperation::CopyRe(math::Matrix* dst, math::Matrix* src, MatrixCopier* matrixCopier, IMathOperation *thiz) {
        if (thiz->m_subcolumns[0] == -1 && thiz->m_subrows[0] == -1) {
            matrixCopier->copyReMatrixToReMatrix(dst, src);
        }
        bool b = !matrixCopier->isError();
        return b;
    }

    bool IMathOperation::IsIm(math::Matrix* matrix, MatrixUtils* matrixUtils) {
        return matrixUtils->isImMatrix(matrix);
    }

    bool IMathOperation::IsRe(math::Matrix* matrix, MatrixUtils* matrixUtils) {
        return matrixUtils->isReMatrix(matrix);
    }

    IMathOperation::IMathOperation(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) : m_matrixModule(_matrixModule),
    m_matrixStructureUtils(_matrixStructureUtils) {
        m_subrows[0] = -1;
        m_subrows[1] = -1;
        m_subcolumns[0] = -1;
        m_subcolumns[1] = -1;
    }

    IMathOperation::~IMathOperation() {
    }

    Status IMathOperation::start() {
        Status status = this->beforeExecution();
        if (status == 0) {
            this->execute();
            status = this->afterExecution();
        }
        return status;
    }

    TwoMatricesOperations::TwoMatricesOperations(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) :
    IMathOperation(_matrixModule, _matrixStructureUtils) {
        this->m_matrix1 = NULL;
        this->m_matrix2 = NULL;
        this->m_output = NULL;
        m_matrixStructure1 = m_matrixStructureUtils->newMatrixStructure();
        m_matrixStructure2 = m_matrixStructureUtils->newMatrixStructure();
        m_outputStructure = m_matrixStructureUtils->newMatrixStructure();
    }

    TwoMatricesOperations::~TwoMatricesOperations() {
        this->m_matrix1 = NULL;
        this->m_matrix2 = NULL;
        this->m_output = NULL;
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure1);
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure2);
        m_matrixStructureUtils->deleteMatrixStructure(m_outputStructure);
    }

    MatrixValueOperation::MatrixValueOperation(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) :
    IMathOperation(_matrixModule, _matrixStructureUtils) {
        this->m_matrix = NULL;
        this->m_output = NULL;
        this->m_revalue = NULL;
        this->m_imvalue = NULL;
        m_matrixStructure = m_matrixStructureUtils->newMatrixStructure();
        m_outputStructure = m_matrixStructureUtils->newMatrixStructure();
    }

    MatrixValueOperation::~MatrixValueOperation() {
        this->m_matrix = NULL;
        this->m_output = NULL;
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure);
        m_matrixStructureUtils->deleteMatrixStructure(m_outputStructure);
    }

    MatrixOperationOutputMatrix::MatrixOperationOutputMatrix(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) :
    IMathOperation(_matrixModule, _matrixStructureUtils) {
        this->m_matrix = NULL;
        this->m_output = NULL;
        m_matrixStructure = m_matrixStructureUtils->newMatrixStructure();
        m_outputStructure = m_matrixStructureUtils->newMatrixStructure();
    }

    MatrixOperationOutputMatrix::~MatrixOperationOutputMatrix() {
        this->m_matrix = NULL;
        this->m_output = NULL;
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure);
        m_matrixStructureUtils->deleteMatrixStructure(m_outputStructure);
    }

    MatrixOperationOutputValue::MatrixOperationOutputValue(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) :
    IMathOperation(_matrixModule, _matrixStructureUtils) {
        this->m_matrix = NULL;
        this->m_output1 = 0;
        m_matrixStructure = m_matrixStructureUtils->newMatrixStructure();
    }

    MatrixOperationOutputValue::~MatrixOperationOutputValue() {
        this->m_matrix = NULL;
        this->m_output1 = 0;
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure);
    }

    MatrixOperationOutputValues::MatrixOperationOutputValues(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) :
    IMathOperation(_matrixModule, _matrixStructureUtils) {
        this->m_matrix = NULL;
        this->m_reoutputs = NULL;
        this->m_imoutputs = NULL;
        this->m_count = 0;
        m_matrixStructure = m_matrixStructureUtils->newMatrixStructure();
    }

    MatrixOperationOutputValues::~MatrixOperationOutputValues() {
        this->m_matrix = NULL;
        this->m_reoutputs = NULL;
        this->m_imoutputs = NULL;
        this->m_count = 0;
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure);
    }

    MatrixOperationTwoOutputs::MatrixOperationTwoOutputs(MatrixModule* _matrixModule,
            MatrixStructureUtils* _matrixStructureUtils) :
    IMathOperation(_matrixModule, _matrixStructureUtils),
    m_matrixStructure(NULL), m_outputStructure1(NULL), m_outputStructure2(NULL) {
        this->m_matrix = NULL;
        this->m_output1 = NULL;
        this->m_output2 = NULL;
        m_matrixStructure = m_matrixStructureUtils->newMatrixStructure();
        m_outputStructure1 = m_matrixStructureUtils->newMatrixStructure();
        m_outputStructure2 = m_matrixStructureUtils->newMatrixStructure();
    }

    MatrixOperationTwoOutputs::~MatrixOperationTwoOutputs() {
        this->m_matrix = NULL;
        this->m_output1 = NULL;
        this->m_output2 = NULL;
        m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure);
        m_matrixStructureUtils->deleteMatrixStructure(m_outputStructure1);
        m_matrixStructureUtils->deleteMatrixStructure(m_outputStructure2);
    }

    void TwoMatricesOperations::setMatrix1(Matrix* matrix) {
        this->m_matrix1 = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_matrixStructure1,
                this->m_matrix1);
    }

    void TwoMatricesOperations::setMatrix2(Matrix* matrix) {
        this->m_matrix2 = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_matrixStructure2,
                this->m_matrix2);
    }

    void TwoMatricesOperations::setOutputMatrix(Matrix* matrix) {
        this->m_output = matrix;
        this->m_matrixStructureUtils->setMatrix(m_outputStructure, matrix);
    }

    void MatrixValueOperation::setMatrix(Matrix* matrix) {
        this->m_matrix = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_matrixStructure, matrix);
    }

    void MatrixValueOperation::setReValue(floatt* value) {
        this->m_revalue = value;
    }

    void MatrixValueOperation::setImValue(floatt* value) {
        this->m_imvalue = value;
    }

    void MatrixValueOperation::setOutputMatrix(Matrix* matrix) {
        this->m_output = matrix;
        this->m_matrixStructureUtils->setMatrix(m_outputStructure, matrix);
    }

    void MatrixOperationOutputMatrix::setMatrix(Matrix* matrix) {
        this->m_matrix = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_matrixStructure, matrix);
    }

    void MatrixOperationOutputMatrix::setOutputMatrix(Matrix* matrix) {
        this->m_output = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_outputStructure, matrix);
    }

    void MatrixOperationOutputValue::setMatrix(Matrix* matrix) {
        this->m_matrix = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_matrixStructure, matrix);
    }

    void MatrixOperationOutputValue::setOutputValue1(floatt* value) {
        this->m_output1 = value;
    }

    void MatrixOperationOutputValue::setOutputValue2(floatt* value) {
        this->m_output2 = value;
    }

    void MatrixOperationOutputValues::setMatrix(Matrix* matrix) {
        this->m_matrix = matrix;
        this->m_matrixStructureUtils->setMatrix(this->m_matrixStructure, matrix);
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
        this->m_matrixStructureUtils->setMatrix(m_matrixStructure, matrix);
    }

    void MatrixOperationTwoOutputs::setOutputMatrix1(Matrix* matrix) {
        this->m_output1 = matrix;
        this->m_matrixStructureUtils->setMatrix(m_outputStructure1, matrix);
    }

    void MatrixOperationTwoOutputs::setOutputMatrix2(Matrix* matrix) {
        this->m_output2 = matrix;
        this->m_matrixStructureUtils->setMatrix(m_outputStructure2, matrix);
    }
}
