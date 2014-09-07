/* 
 * File:   MatrixOperationsCPU.h
 * Author: mmatula
 *
 * Created on September 24, 2013, 9:33 PM
 */

#ifndef OGLA_MATRIXOPERATIONSCPU_H
#define	OGLA_MATRIXOPERATIONSCPU_H

#include "ThreadsCpu.h"   

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
        math::Matrix* R1;
        math::Matrix* Q1;
        math::Matrix* G;
        math::Matrix* GT;
        DotProductOperationCpu dotProduct;
        TransposeOperationCpu transpose;
        inline void prepareGMatrix(math::Matrix* A,
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

    class MathOperationsCpu : public utils::Module {
#ifdef DEBUG
        std::map<void*, std::string> valuesNames;
#endif
        int threadsCount;
        int serieLimit;
        AdditionOperationCpu additionOperation;
        SubstracionOperationCpu substracionOperation;
        DotProductOperationCpu dotProductOperation;
        TensorProductOperationCpu tensorProductOperation;
        DiagonalizationOperationCpu diagonalizationOperation;
        ExpOperationCpu expOperation;
        MultiplicationConstOperationCpu multiplicationConstOperation;
        MagnitudeOperationCpu magnitudeOperation;
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
        void setSubRows(intt begin, intt end);
        void setSubColumns(intt begin, intt end);
        void unsetSubRows();
        void unsetSubColumns();
        math::Status add(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2);
        math::Status substract(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2);
        math::Status dotProduct(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2);
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
        math::Status magnitude(floatt* output, math::Matrix* matrix1);
        math::Status transpose(math::Matrix* output,
                math::Matrix* matrix1);
        math::Status transpose(math::Matrix* matrix);
        math::Status det(floatt* output, math::Matrix* matrix);
        math::Status det(floatt* output,
                floatt* output1, math::Matrix* matrix);
        math::Status qrDecomposition(math::Matrix* Q,
                math::Matrix* R, math::Matrix* matrix);
    };

}
#endif	/* MATRIXOPERATIONSCPU_H */

