/* 
 * File:   math::MatrixOperationsCPU.cpp
 * Author: mmatula
 * 
 * Created on September 24, 2013, 9:33 PM
 */
#include <vector>

#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "Matrix.h"
#include "ThreadUtils.h"
#include "ThreadsMapper.h"
#include "HostMatrixStructure.h"
#include <math.h>

#define ReIsNotNull(m) m->reValues != NULL
#define ImIsNotNull(m) m->imValues != NULL

inline void printInfo(const char* function,
        const math::Matrix* output, const math::Matrix* arg1, const math::Matrix* arg2) {
    std::string temp;
    fprintf(stdout, "Function: %s\n", function);
    host::GetReMatrixStr(temp, output);
    fprintf(stdout, "output = %s\n", temp.c_str());
    host::GetReMatrixStr(temp, arg1);
    fprintf(stdout, "arg1 = %s\n", temp.c_str());
    host::GetReMatrixStr(temp, arg2);
    fprintf(stdout, "arg2 = %s\n\n", temp.c_str());
}

inline void printInfo(const char* function, const math::Matrix* output, const math::Matrix* arg1,
        floatt* arg2) {
    std::string temp;
    fprintf(stdout, "Function: %s\n", function);
    host::GetReMatrixStr(temp, output);
    fprintf(stdout, "output = %s\n", temp.c_str());
    host::GetReMatrixStr(temp, arg1);
    fprintf(stdout, "arg1 = %s\n", temp.c_str());
    fprintf(stdout, "arg2 = %f\n\n", *arg2);
}

inline void printInfo(const char* function, const math::Matrix* output, const math::Matrix* arg1) {
    std::string temp;
    fprintf(stdout, "Function: %s\n", function);
    host::GetReMatrixStr(temp, output);
    fprintf(stdout, "output = %s\n", temp.c_str());
    host::GetReMatrixStr(temp, arg1);
    fprintf(stdout, "arg1 = %s\n\n", temp.c_str());
}

inline void printInfo(const char* function, floatt* output, const math::Matrix* arg1, floatt* arg2) {
    std::string temp;
    fprintf(stdout, "Function: %s\n", function);
    fprintf(stdout, "output = %s\n", temp.c_str());
    host::GetReMatrixStr(temp, arg1);
    fprintf(stdout, "arg1 = %s\n", temp.c_str());
    fprintf(stdout, "arg2 = %f\n\n", *arg2);
}

inline void printInfo(const char* function, floatt* output, const math::Matrix* arg1) {
    std::string temp;
    fprintf(stdout, "Function: %s\n", function);
    fprintf(stdout, "output = %f\n", *output);
    host::GetReMatrixStr(temp, arg1);
    fprintf(stdout, "arg1 = %s\n\n", temp.c_str());
}

inline void printInfo(const char* function, const math::Matrix* output, const std::string& arg1) {
    std::string temp;
    fprintf(stdout, "Function: %s \n", function);
    host::GetReMatrixStr(temp, output);
    fprintf(stdout, "output = %s \n", temp.c_str());
    fprintf(stdout, "arg1 = %s \n\n", arg1.c_str());
}

namespace math {
    namespace cpu {

        ThreadsCountProperty::ThreadsCountProperty() : m_threadsCount(1) {
        }

        ThreadsCountProperty::~ThreadsCountProperty() {
        }

        void ThreadsCountProperty::setThreadsCount(uintt threadsCount) {
            m_threadsCount = threadsCount;
        }

        MathOperations::MathOperations() : utils::Module() {
            m_subrows[0] = 0;
            m_subrows[1] = 0;
            m_subcolumns[0] = 0;
            m_subcolumns[1] = 0;
            registerMathOperation(&additionOperation);
            registerMathOperation(&substracionOperation);
            registerMathOperation(&dotProductOperation);
            registerMathOperation(&tensorProductOperation);
            registerMathOperation(&diagonalizationOperation);
            registerMathOperation(&expOperation);
            registerMathOperation(&multiplicationConstOperation);
            registerMathOperation(&magnitudeOperation);
            registerMathOperation(&m_transposeOperation);
        }

        void MathOperations::registerMathOperation(IMathOperation* mathOperation) {
            operations.push_back(mathOperation);
            registerThreadsCountProperty(mathOperation);
        }

        void MathOperations::registerThreadsCountProperty(IMathOperation* mathOperation) {
            ThreadsCountProperty* threadsCountProperty =
                    dynamic_cast<ThreadsCountProperty*> (mathOperation);
            if (threadsCountProperty) {
                properties.push_back(threadsCountProperty);
            }
        }

        MathOperations::~MathOperations() {
        }

        math::Status MathOperations::execute(math::TwoMatricesOperations& obj,
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
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        math::Status MathOperations::execute(math::MatrixValueOperation& obj,
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
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        math::Status MathOperations::execute(math::MatrixValueOperation& obj,
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
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        math::Status MathOperations::execute(math::MatrixOperationOutputMatrix& obj,
                math::Matrix* output, math::Matrix* arg1) {
            obj.setSubRows(m_subrows);
            obj.setSubColumns(m_subcolumns);
            obj.setOutputMatrix(output);
            obj.setMatrix(arg1);
            math::Status status = obj.start();
            unsetSubRows();
            unsetSubColumns();
            if (status != 0) {
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        math::Status MathOperations::execute(math::MatrixOperationOutputValue& obj,
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
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        math::Status MathOperations::execute(math::MatrixOperationOutputValue& obj,
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
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        math::Status MathOperations::execute(math::MatrixOperationTwoOutputs& obj,
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
                this->addMessageLine(getErrorStr(status));
            }
            return status;
        }

        void MathOperations::setThreadsCount(int threadsCount) {
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

        void MathOperations::setSerieLimit(int serieLimit) {
            this->serieLimit = serieLimit;
        }

        void MathOperations::setSubRows(intt begin, intt end) {
            m_subrows[0] = begin;
            m_subrows[1] = end;
        }

        void MathOperations::setSubColumns(intt begin, intt end) {
            m_subcolumns[0] = begin;
            m_subcolumns[1] = end;
        }

        void MathOperations::unsetSubRows() {
            this->m_subrows[0] = 0;
            this->m_subrows[1] = 0;
            for (unsigned int fa = 0; fa < operations.size(); fa++) {
                operations[fa]->unsetSubRows();
            }
        }

        void MathOperations::unsetSubColumns() {
            this->m_subcolumns[0] = 0;
            this->m_subcolumns[1] = 0;
            for (unsigned int fa = 0; fa < operations.size(); fa++) {
                operations[fa]->unsetSubColumns();
            }
        }

        void MathOperations::registerValueName(void* value, const std::string& name) {
#ifdef DEBUG_MATRIX_OPERATIONS
            valuesNames[value] = name;
#endif
        }

        math::Status MathOperations::add(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2) {
            math::Status status = execute(this->additionOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
            return status;
        }

        math::Status MathOperations::substract(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2) {
            math::Status status = execute(this->substracionOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
            return status;
        }

        math::Status MathOperations::dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2) {
            math::Status status = execute(this->dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
            return status;
        }

        math::Status MathOperations::tensorProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2) {
            math::Status status = execute(this->tensorProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
            return status;
        }

        math::Status MathOperations::diagonalize(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2) {
            math::Status status = execute(this->diagonalizationOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
            return status;
        }

        math::Status MathOperations::multiply(math::Matrix* output, math::Matrix* matrix1, floatt* value) {
            math::Status status = execute(this->multiplicationConstOperation, output, matrix1, value);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, value);
#endif
            return status;
        }

        math::Status MathOperations::multiply(math::Matrix* output,
                math::Matrix* matrix1, floatt* revalue, floatt* imvalue) {
            math::Status status = execute(this->multiplicationConstOperation,
                    output, matrix1, revalue, imvalue);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, value);
#endif
            return status;
        }

        math::Status MathOperations::exp(math::Matrix* output, math::Matrix* matrix1) {
            math::Status status = execute(this->expOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1);
#endif
            return status;
        }

        math::Status MathOperations::multiply(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2) {
            math::Status status = execute(this->dotProductOperation, output, matrix1, matrix2);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1, matrix2);
#endif
            return status;
        }

        math::Status MathOperations::magnitude(floatt* output, math::Matrix* matrix1) {
            math::Status status = execute(magnitudeOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1);
#endif
            return status;
        }

        math::Status MathOperations::transpose(math::Matrix* output, math::Matrix* matrix1) {
            math::Status status = execute(m_transposeOperation, output, matrix1);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, output, matrix1);
#endif
            return status;
        }

        math::Status MathOperations::transpose(math::Matrix* matrix) {
#ifdef DEBUG_MATRIX_OPERATIONS            
            std::string matrixStr = "";
            host::GetReMatrixStr(matrixStr, matrix);
#endif
            math::Status status = execute(m_transposeOperation, matrix, matrix);
#ifdef DEBUG_MATRIX_OPERATIONS
            printInfo(__FUNCTION__, matrix, matrixStr);
#endif
            return status;
        }

        math::Status MathOperations::det(floatt* output, math::Matrix* matrix) {
            return execute(m_determinantOperation, output, matrix);
        }

        math::Status MathOperations::det(floatt* output, floatt* output1, math::Matrix* matrix) {
            return execute(m_determinantOperation, output, output1, matrix);
        }

        math::Status MathOperations::qrDecomposition(math::Matrix* Q,
                math::Matrix* R, math::Matrix* matrix) {
            return execute(m_qrDecomposition, Q, R, matrix);
        }



#define GET(x,y,index) x+index*y 
#define DEFAULT_CONSTRUCTOR(cname) cname::cname():math::I##cname(&(HostMatrixModules::GetInstance()),\
        HostMatrixStructureUtils::GetInstance(&(HostMatrixModules::GetInstance()))){} cname::~cname(){}
#define DEFAULT_CONSTRUCTOR_WITH_ARGS(cname,code) cname::cname():math::I##cname(){code} cname::~cname(){}

#define CHECK_PARAMS_PTR() if(this->output==NULL || matrix1 == NULL || matrix2 == NULL){return STATUS_INVALID_PARAMS;}
#define CHECK_PARAMS_PTR_1() if(this->output==NULL || matrix1 == NULL){return STATUS_INVALID_PARAMS;}
#define CHECK_PARAMS_PTR_3(a,b,c) if(a==NULL || b == NULL || c == NULL){return STATUS_INVALID_PARAMS;}

        DEFAULT_CONSTRUCTOR(AdditionOperation);
        DEFAULT_CONSTRUCTOR(SubstracionOperation)
        DEFAULT_CONSTRUCTOR(DotProductOperation);
        DEFAULT_CONSTRUCTOR(TensorProductOperation);
        DEFAULT_CONSTRUCTOR(MagnitudeOperation);
        DEFAULT_CONSTRUCTOR(DiagonalizationOperation);
        DEFAULT_CONSTRUCTOR(MultiplicationConstOperation);
        DEFAULT_CONSTRUCTOR(ExpOperation);
        //        DEFAULT_CONSTRUCTOR(QRDecomposition);
        DEFAULT_CONSTRUCTOR(TransposeOperation);


        std::vector<void*> ptrs;

        extern "C" {

            void GetModulesCount(uint& count) {

                count = 7;
            }

            void LoadModules(utils::OglaObject** objects) {
                /*
                objects[0] = new math::ModuleCreatorCPU("Addition", new math::cpu::AdditionOperation());
                objects[1] = new math::ModuleCreatorCPU("Substraction", new math::cpu::SubstracionOperation());
                objects[2] = new math::ModuleCreatorCPU("Multiplication", new math::cpu::DotProductOperation());
                objects[3] = new math::ModuleCreatorCPU("MultiplicationDC", new math::cpu::DotProductDC());
                objects[4] = new math::ModuleCreatorCPU("ConstantMultiplication", new math::cpu::MultiplicationConstOperation());
                objects[5] = new math::ModuleCreatorCPU("Exp", new math::cpu::ExpOperation());
                objects[6] = new math::ModuleCreatorCPU("ExpDC", new math::cpu::ExpOperationDC());
             
                for (int fa = 0; fa < 6; fa++) {
                    ptrs.push_back(objects[fa]);
                }*/
            }

            void UnloadModule(utils::OglaObject* object) {
                /*math::ModuleCreatorCPU* moduleCreatorCPU = dynamic_cast<math::ModuleCreatorCPU*> (object);
                if (moduleCreatorCPU != NULL) {
                    std::vector<void*>::iterator it = std::find(ptrs.begin(), ptrs.end(), (void*) object);
                    if (it != ptrs.end()) {
                        delete moduleCreatorCPU;
                        ptrs.erase(it);
                    }
                }*/
            }
        }
    }
}