/*
 * Copyright 2016 - 2021 Marcin Matula
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

#include "Matrix.h"
#include "Math.h"

#include <string>

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

    class IMathOperation  {
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
        static bool CopyIm(math::ComplexMatrix* dst, math::ComplexMatrix* src, IMathOperation *thiz);
        static bool CopyRe(math::ComplexMatrix* dst, math::ComplexMatrix* src, IMathOperation *thiz);
        static bool IsIm(math::ComplexMatrix* matrix);
        static bool IsRe(math::ComplexMatrix* matrix);
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
        IMathOperation();
        virtual ~IMathOperation();
        Status start();
    };

    class TwoMatricesOperations : public IMathOperation {
    protected:
        TwoMatricesOperations();
        virtual ~TwoMatricesOperations();
        ComplexMatrix* m_matrix1;
        ComplexMatrix* m_matrix2;
        ComplexMatrix* m_output;
        Status beforeExecution();
        Status afterExecution();
    public:
        void setMatrix1(ComplexMatrix* matrix);
        void setMatrix2(ComplexMatrix* matrix);
        void setOutputMatrix(ComplexMatrix* matrix);
    };

    class MatrixValueOperation : public IMathOperation {
    protected:
        ComplexMatrix* m_matrix;
        ComplexMatrix* m_output;
        floatt* m_revalue;
        floatt* m_imvalue;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixValueOperation();
        virtual ~MatrixValueOperation();
        void setMatrix(ComplexMatrix* matrix);
        void setReValue(floatt* value);
        void setImValue(floatt* value);
        void setOutputMatrix(ComplexMatrix* matrix);
    };

    class MatrixOperationOutputMatrix : public IMathOperation {
    protected:
        ComplexMatrix* m_matrix;
        ComplexMatrix* m_output;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationOutputMatrix();
        virtual ~MatrixOperationOutputMatrix();
        void setMatrix(ComplexMatrix* matrix);
        void setOutputMatrix(ComplexMatrix* matrix);
    };

    class MatrixOperationOutputValue : public IMathOperation {
    protected:
        ComplexMatrix* m_matrix;
        floatt* m_output1;
        floatt* m_output2;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationOutputValue();
        virtual ~MatrixOperationOutputValue();
        void setMatrix(ComplexMatrix* matrix);
        void setOutputValue1(floatt* value);
        void setOutputValue2(floatt* value);
    };

    class MatrixOperationOutputValues : public IMathOperation {
    protected:
        ComplexMatrix* m_matrix;
        floatt* m_reoutputs;    
        floatt* m_imoutputs;
        uintt m_count;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationOutputValues();
        virtual ~MatrixOperationOutputValues();
        void setMatrix(ComplexMatrix* matrix);
        void setReOutputValues(floatt* revalue, uintt count);
        void setImOutputValues(floatt* imvalue, uintt count);
    };

    class MatrixOperationTwoOutputs : public IMathOperation {
    protected:
        ComplexMatrix* m_matrix;
        ComplexMatrix* m_output1;
        ComplexMatrix* m_output2;
        Status beforeExecution();
        Status afterExecution();
    public:
        MatrixOperationTwoOutputs();
        virtual ~MatrixOperationTwoOutputs();
        void setMatrix(ComplexMatrix* matrix);
        void setOutputMatrix1(ComplexMatrix* matrix);
        void setOutputMatrix2(ComplexMatrix* matrix);
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
        Status beforeExecution(math::ComplexMatrix* output,
                math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                bool(*HasInstance)(math::ComplexMatrix* matrix),
                ExecutionPath& executionPath);
    public:
        IAdditionOperation();
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
        Status beforeExecution(math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                bool(*copy)(math::ComplexMatrix* src, math::ComplexMatrix* dst, math::IMathOperation* thiz),
                bool(*isNotNull)(math::ComplexMatrix* matrix),
                ISubstracionOperation::ExecutionPath& executionPath);
    public:
        ISubstracionOperation();
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
        Status beforeExecution(math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                bool(*copy)(math::ComplexMatrix* src, math::ComplexMatrix* dst, math::IMathOperation* thiz),
                bool(*isNotNull)(math::ComplexMatrix* matrix),
                IDotProductOperation::ExecutionPath& executionPath);
    public:
        IDotProductOperation();
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
        Status prepare(math::ComplexMatrix* output, math::ComplexMatrix* matrix1, floatt* value,
                bool(*copy)(math::ComplexMatrix* src, math::ComplexMatrix* dst,
                math::IMathOperation* thiz),
                bool(*isNotNull)(math::ComplexMatrix* matrix),
                IMultiplicationConstOperation::ExecutionPath& executionPath);
    public:
        IMultiplicationConstOperation();
        virtual ~IMultiplicationConstOperation();
    };

    class IExpOperation : public MatrixOperationOutputMatrix {
    protected:
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IExpOperation();
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
        Status beforeExecution(math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                bool(*copy)(math::ComplexMatrix* src, math::ComplexMatrix* dst, math::IMathOperation* thiz),
                bool(*isNotNull)(math::ComplexMatrix* matrix),
                IDiagonalizationOperation::ExecutionPath& executionPath);
    public:
        IDiagonalizationOperation();
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
        Status beforeExecution(math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                bool(*copy)(math::ComplexMatrix* src, math::ComplexMatrix* dst, math::IMathOperation* thiz),
                bool(*isNotNull)(math::ComplexMatrix* matrix),
                ITensorProductOperation::ExecutionPath& executionPath);
    public:
        ITensorProductOperation();
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
        IMagnitudeOperation();
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
        ITransposeOperation();
        virtual ~ITransposeOperation();
    };

    class IDeterminantOperation : public MatrixOperationOutputValue {

        enum ExecutionPath {
            EXECUTION_NORMAL,
            EXECUTION_NOTHING
        };

        Status prepare(floatt* output, math::ComplexMatrix* matrix,
                bool(*isNotNull)(math::ComplexMatrix* matrix),
                ExecutionPath& executionPath);

        ExecutionPath m_executionPathRe;
        ExecutionPath m_executionPathIm;
    protected:
        Status beforeExecution();
        virtual void execute() = 0;
    public:
        IDeterminantOperation();
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
        IQRDecomposition();
        virtual ~IQRDecomposition();
    };
}

#endif	/* MATHOPERATIONS_H */
