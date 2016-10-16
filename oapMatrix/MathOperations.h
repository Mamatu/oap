/*
 * Copyright 2016 Marcin Matula
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



#ifndef OAP_MATH_OPERATIONS_H
#define	OAP_MATH_OPERATIONS_H

#include "MatrixModules.h"

namespace math {

    enum Status {
        STATUS_OK,
        STATUS_INVALID_PARAMS,
        STATUS_ERROR,
        STATUS_NOT_SUPPORTED_SUBMATRIX,
        STATUS_NOT_SUPPORTED
    };

    const char* getStr(Status status);
    const char* getErrorStr(Status status);

    class IMathOperation : public utils::Module {
    protected:
        /**
         * \brief Contains information about user specified 
         *        range of rows. 
         * 
         * \details m_subrows[0] is begin row,
         *          m_subrows[1] is the next row after last row.
         *          m_subrows[1] - is default supported, 
         *          m_subrows[0] can be supported by procedure's implementation.
         */
        uintt m_subrows[2];
        
        /**
         * \brief Contains information about user specified 
         *        range of rows. 
         * 
         * \details m_subcolumns[0] is begin row,
         *          m_subcolumns[1] is the next row after last row.
         *          m_subcolumns[1] - is default supported, 
         *          m_subcolumns[0] can be supported by procedure's implementation.
         */
        uintt m_subcolumns[2];
        
        /**
         * \brief Pointer to value
         */
        MatrixModule* m_module;
        static bool CopyIm(math::Matrix* dst, math::Matrix* src,
                MatrixCopier* matrixCopier, IMathOperation *thiz);
        static bool CopyRe(math::Matrix* dst, math::Matrix* src,
                MatrixCopier* matrixCopier, IMathOperation *thiz);
        static bool IsIm(math::Matrix* matrix, MatrixUtils* matrixUtils);
        static bool IsRe(math::Matrix* matrix, MatrixUtils* matrixUtils);
        virtual void execute() = 0;
        virtual Status beforeExecution() = 0;
        virtual Status afterExecution() = 0;
    public:
        void setSubRows(uintt subrows);
        void setSubColumns(uintt subcolumns);
        void setSubRows(uintt subrows[2]);
        void setSubColumns(uintt subcolumns[2]);
        void unsetSubRows();
        void unsetSubColumns();
        IMathOperation(MatrixModule* matrixModule);
        virtual ~IMathOperation();
        Status start();
    };

    class TwoMatricesOperations : public IMathOperation {
    protected:
        TwoMatricesOperations(MatrixModule* matrixModule);
        virtual ~TwoMatricesOperations();
        Matrix* m_matrix1;
        Matrix* m_matrix2;
        Matrix* m_output;
        Status beforeExecution();
        Status afterExecution();
    public:
        void setMatrix1(Matrix* matrix);
        void setMatrix2(Matrix* matrix);
        void setOutputMatrix(Matrix* matrix);
    };

    class MatrixValueOperation : public IMathOperation {
    protected:
        Matrix* m_matrix;
        Matrix* m_output;
        floatt* m_revalue;
        floatt* m_imvalue;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixValueOperation(MatrixModule* matrixModule);
        virtual ~MatrixValueOperation();
        void setMatrix(Matrix* matrix);
        void setReValue(floatt* value);
        void setImValue(floatt* value);
        void setOutputMatrix(Matrix* matrix);
    };

    class MatrixOperationOutputMatrix : public IMathOperation {
    protected:
        Matrix* m_matrix;
        Matrix* m_output;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationOutputMatrix(MatrixModule* matrixModule);
        virtual ~MatrixOperationOutputMatrix();
        void setMatrix(Matrix* matrix);
        void setOutputMatrix(Matrix* matrix);
    };

    class MatrixOperationOutputValue : public IMathOperation {
    protected:
        Matrix* m_matrix;
        floatt* m_output1;
        floatt* m_output2;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationOutputValue(MatrixModule* matrixModule);
        virtual ~MatrixOperationOutputValue();
        void setMatrix(Matrix* matrix);
        void setOutputValue1(floatt* value);
        void setOutputValue2(floatt* value);
    };

    class MatrixOperationOutputValues : public IMathOperation {
    protected:
        Matrix* m_matrix;
        floatt* m_reoutputs;    
        floatt* m_imoutputs;
        uintt m_count;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationOutputValues(MatrixModule* matrixModule);
        virtual ~MatrixOperationOutputValues();
        void setMatrix(Matrix* matrix);
        void setReOutputValues(floatt* revalue, uintt count);
        void setImOutputValues(floatt* imvalue, uintt count);
    };

    class MatrixOperationTwoOutputs : public IMathOperation {
    protected:
        Matrix* m_matrix;
        Matrix* m_output1;
        Matrix* m_output2;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationTwoOutputs(MatrixModule* matrixModule);
        virtual ~MatrixOperationTwoOutputs();
        void setMatrix(Matrix* matrix);
        void setOutputMatrix1(Matrix* matrix);
        void setOutputMatrix2(Matrix* matrix);
    };

    class IAdditionOperation : public TwoMatricesOperations {
    protected:

        enum ExecutionPath {
            EXECUTION_NOTHING,
            EXECUTION_NORMAL,
            EXECUTION_OUTPUT_TO_ZEROS,
            EXECUTION_COPY_FIRST_PARAM,
            EXECUTION_COPY_SECOND_PARAM
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    private:
        Status beforeExecution(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2,
                bool(*HasInstance)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                ExecutionPath& executionPath);
    public:
        IAdditionOperation(MatrixModule* matrixModule);
        virtual ~IAdditionOperation();
    };

    class ISubstracionOperation : public TwoMatricesOperations {
    protected:

        enum ExecutionPath {
            EXECUTION_NOTHING,
            EXECUTION_NORMAL,
            EXECUTION_MULTIPLY_BY_MINUS_ONE,
            EXECUTION_OUTPUT_TO_ZEROS
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    private:
        Status beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
                bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                ISubstracionOperation::ExecutionPath& executionPath);
    public:
        ISubstracionOperation(MatrixModule* matrixModule);
        virtual ~ISubstracionOperation();
    };

    class IDotProductOperation : public TwoMatricesOperations {
    protected:

        enum ExecutionPath {
            EXECUTION_NOTHING,
            EXECUTION_NORMAL,
            EXECUTION_OUTPUT_TO_ZEROS
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;

        uintt m_offset[2];

        Status beforeExecution();
        Status afterExecution();
        virtual void execute() = 0;
    private:
        Status beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
                bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                IDotProductOperation::ExecutionPath& executionPath);
    public:
        IDotProductOperation(MatrixModule* matrixModule);
        virtual ~IDotProductOperation();
        void setOffset(uintt offset);
        void setOffset(uintt boffset, uintt eoffset);
    };

    class IMultiplicationConstOperation : public MatrixValueOperation {
    protected:

        enum ExecutionPath {
            EXECUTION_NOTHING,
            EXECUTION_NORMAL,
            EXECUTION_ZEROS_TO_OUTPUT
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    private:
        Status prepare(math::Matrix* output, math::Matrix* matrix1, floatt* value,
                bool(*copy)(math::Matrix* src, math::Matrix* dst,
                MatrixCopier* matrixCopier,
                math::IMathOperation* thiz),
                bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                IMultiplicationConstOperation::ExecutionPath& executionPath);
    public:
        IMultiplicationConstOperation(MatrixModule* matrixModule);
        virtual ~IMultiplicationConstOperation();
    };

    class IExpOperation : public MatrixOperationOutputMatrix {
    protected:
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IExpOperation(MatrixModule* matrixModule);
        virtual ~IExpOperation();
    };

    class IDiagonalizationOperation : public TwoMatricesOperations {
    protected:

        enum ExecutionPath {
            EXECUTION_NOTHING,
            EXECUTION_NORMAL,
            EXECUTION_ZEROS_TO_OUTPUT
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    private:
        Status beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
                bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                IDiagonalizationOperation::ExecutionPath& executionPath);
    public:
        IDiagonalizationOperation(MatrixModule* matrixModule);
        virtual ~IDiagonalizationOperation();
    };

    class ITensorProductOperation : public TwoMatricesOperations {
    protected:

        enum ExecutionPath {
            EXECUTION_NOTHING,
            EXECUTION_NORMAL,
            EXECUTION_ZEROS_TO_OUTPUT
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    private:
        Status beforeExecution(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                bool(*copy)(math::Matrix* src, math::Matrix* dst, MatrixCopier* matrixCopier, math::IMathOperation* thiz),
                bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                ITensorProductOperation::ExecutionPath& executionPath);
    public:
        ITensorProductOperation(MatrixModule* matrixModule);
        virtual ~ITensorProductOperation();
    };

    class IMagnitudeOperation : public MatrixOperationOutputValue {
    protected:

        enum ExecutionPath {
            EXECUTION_NORMAL,
            EXECUTION_IS_ZERO
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IMagnitudeOperation(MatrixModule* matrixModule);
        virtual ~IMagnitudeOperation();
    };

    class ITransposeOperation : public MatrixOperationOutputMatrix {
    protected:
        enum ExecutionPath {
            EXECUTION_NORMAL,
            EXECUTION_NOTHING
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        ITransposeOperation(MatrixModule* matrixModule);
        virtual ~ITransposeOperation();
    };

    class IDeterminantOperation : public MatrixOperationOutputValue {

        enum ExecutionPath {
            EXECUTION_NORMAL,
            EXECUTION_NOTHING
        };

        Status prepare(floatt* output, math::Matrix* matrix,
                bool(*isNotNull)(math::Matrix* matrix, MatrixUtils* matrixUtils),
                ExecutionPath& executionPath);

        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
    protected:
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IDeterminantOperation(MatrixModule* matrixModule);
        virtual ~IDeterminantOperation();
    };

    class IQRDecomposition : public MatrixOperationTwoOutputs {
    protected:

        enum ExecutionPath {
            EXECUTION_NORMAL,
            EXECUTION_NOTHING
        };
        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IQRDecomposition(MatrixModule* matrixModule);
        virtual ~IQRDecomposition();
    };
}

#endif	/* MATHOPERATIONS_H */
