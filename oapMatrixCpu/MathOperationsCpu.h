/*
 * Copyright 2016 - 2018 Marcin Matula
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




#ifndef OAP_MATRIXOPERATIONSCPU_H
#define	OAP_MATRIXOPERATIONSCPU_H

#include "ThreadsCpu.h"   
#include <map> 
#include <vector> 

namespace math {

class AdditionOperationCpu :
public IAdditionOperation,
public ThreadsCPU<AdditionOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    AdditionOperationCpu();
    ~AdditionOperationCpu();
};

class SubstracionOperationCpu :
public ISubstracionOperation,
public ThreadsCPU<SubstracionOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    SubstracionOperationCpu();
    ~SubstracionOperationCpu();
};

class DotProductOperationCpu :
public IDotProductOperation,
public ThreadsCPU<DotProductOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    DotProductOperationCpu();
    ~DotProductOperationCpu();
};

class MultiplicationConstOperationCpu :
public IMultiplicationConstOperation,
public ThreadsCPU<MultiplicationConstOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    MultiplicationConstOperationCpu();
    ~MultiplicationConstOperationCpu();
};

class ExpOperationCpu :
public IExpOperation,
public ThreadsCountProperty {
    uintt serieLimit;
protected:
    DotProductOperationCpu dotProduct;
    MultiplicationConstOperationCpu multiplication;
    AdditionOperationCpu addition;
public:
    void execute();
    ExpOperationCpu();
    ~ExpOperationCpu();
};

class DiagonalizationOperationCpu :
public IDiagonalizationOperation,
public ThreadsCPU<DiagonalizationOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    DiagonalizationOperationCpu();
    virtual ~DiagonalizationOperationCpu();
};

class TensorProductOperationCpu :
public ITensorProductOperation,
public ThreadsCPU<TensorProductOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    TensorProductOperationCpu();
    ~TensorProductOperationCpu();
};

class MagnitudeOperationCpu :
public IMagnitudeOperation,
public ThreadsCPU<MagnitudeOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    MagnitudeOperationCpu();
    ~MagnitudeOperationCpu();
};

class TransposeOperationCpu :
public ITransposeOperation,
public ThreadsCPU<TransposeOperationCpu> {
    static void Execute(void* ptr);
protected:
    void execute();
public:
    TransposeOperationCpu();
    ~TransposeOperationCpu();
};

class QRDecompositionCpu :
public IQRDecomposition,
public ThreadsCountProperty {
    static void Execute(void* ptr);
    math::Matrix* m_R1;
    math::Matrix* m_Q1;
    math::Matrix* m_G;
    math::Matrix* m_GT;
    DotProductOperationCpu dotProduct;
    TransposeOperationCpu transpose;
    inline void prepareGMatrix(math::Matrix* Matrix,
        uintt column, uintt row,
        math::Matrix* G);
protected:
    void execute();
public:
    QRDecompositionCpu();
    ~QRDecompositionCpu();
};

class DeterminantOperationCpu :
public IDeterminantOperation,
public ThreadsCountProperty {
    QRDecompositionCpu m_qrDecomposition;
    math::Matrix* m_q;
    math::Matrix* m_r;
protected:
    math::Status beforeExecution();
    void execute();
public:
    DeterminantOperationCpu();
    ~DeterminantOperationCpu();
};

class MathOperationsCpu;

class MathOperationsCpu  {
#ifdef DEBUG
    std::map<void*, std::string> valuesNames;
#endif
    int threadsCount;
    int serieLimit;
    AdditionOperationCpu m_additionOperation;
    SubstracionOperationCpu m_substracionOperation;
    DotProductOperationCpu m_dotProductOperation;
    TensorProductOperationCpu m_tensorProductOperation;
    DiagonalizationOperationCpu m_diagonalizationOperation;
    ExpOperationCpu m_expOperation;
    MultiplicationConstOperationCpu m_multiplicationConstOperation;
    MagnitudeOperationCpu m_magnitudeOperation;
    TransposeOperationCpu m_transposeOperation;
    DeterminantOperationCpu m_determinantOperation;
    QRDecompositionCpu m_qrDecomposition;
    uintt m_subcolumns[2];
    uintt m_subrows[2];
    void registerMathOperation(IMathOperation* mathOperation);
    void registerThreadsCountProperty(IMathOperation* mathOperation);
    std::vector<IMathOperation*> operations;
    std::vector<ThreadsCountProperty*> properties;

    inline math::Status execute(math::TwoMatricesOperations& obj,
        math::Matrix* output, math::Matrix* arg1, math::Matrix* arg2);

    inline math::Status execute(math::MatrixValueOperation& obj,
        math::Matrix* output, math::Matrix* arg1, floatt* value);

    inline math::Status execute(math::MatrixValueOperation& obj,
        math::Matrix* output, math::Matrix* arg1, floatt* revalue,
        floatt* imvalue);

    inline math::Status execute(math::MatrixOperationOutputMatrix& obj,
        math::Matrix* output, math::Matrix* arg1);

    inline math::Status execute(math::MatrixOperationOutputValue& obj,
        floatt* output, math::Matrix* arg1);

    inline math::Status execute(math::MatrixOperationOutputValue& obj,
        floatt* output1, floatt* output2, math::Matrix* arg1);

    inline math::Status execute(math::MatrixOperationTwoOutputs& obj,
        math::Matrix* output1, math::Matrix* output2,
        math::Matrix* arg1);
public:
    MathOperationsCpu();
    virtual ~MathOperationsCpu();
    void registerValueName(void* value, const std::string& name);
    void setThreadsCount(int threadsCount);
    void setSerieLimit(int serieLimit);
    void setSubColumns(uintt subcolumns);
    void setSubRows(uintt subrows);
    void setSubColumns(uintt subcolumns[2]);
    void setSubRows(uintt subrows[2]);
    void setSubColumns(uintt begin, uintt end);
    void setSubRows(uintt begin, uintt end);
    void unsetSubRows();
    void unsetSubColumns();

    math::Status add(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2);

    math::Status substract(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2);

    math::Status dotProduct(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2);

    math::Status dotProduct(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2, uintt offset);

    math::Status dotProduct(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2,
        uintt boffset, uintt eoffset);

    math::Status tensorProduct(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2);

    math::Status diagonalize(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2);

    math::Status multiply(math::Matrix* output,
        math::Matrix* matrix1, floatt* value);

    math::Status multiply(math::Matrix* output,
        math::Matrix* matrix1, floatt* revalue, floatt* imvalue);

    math::Status exp(math::Matrix* output,
        math::Matrix* matrix1);

    math::Status multiply(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2);

    math::Status multiply(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2, uintt offset);

    math::Status multiply(math::Matrix* output,
        math::Matrix* matrix1, math::Matrix* matrix2,
        uintt boffset, uintt eoffset);

    math::Status magnitude(floatt* output, math::Matrix* matrix1);

    math::Status transpose(math::Matrix* output, math::Matrix* matrix1);
    math::Status transpose(math::Matrix* output, math::Matrix* matrix1,
        uintt subcolumns[2], uintt subrows[2]);

    math::Status transpose(math::Matrix* matrix);

    math::Status det(floatt* output, math::Matrix* matrix);

    math::Status det(floatt* output,
        floatt* output1, math::Matrix* matrix);

    math::Status qrDecomposition(math::Matrix* Q,
        math::Matrix* R, math::Matrix* matrix);
};

}
#endif	/* MATRIXOPERATIONSCPU_H */
