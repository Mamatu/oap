/* 
 * File:   MatrixOperationsCPU.h
 * Author: mmatula
 *
 * Created on September 24, 2013, 9:33 PM
 */

#ifndef OGLA_MATRIXOPERATIONSCPU_H
#define	OGLA_MATRIXOPERATIONSCPU_H

#include "MathOperations.h"
#include "ThreadUtils.h"
#include "HostMatrixModules.h"
#include "Internal.h"   

namespace math {
    namespace cpu {

        class ThreadsCountProperty {
        protected:
            uintt m_threadsCount;
        public:
            ThreadsCountProperty();
            virtual ~ThreadsCountProperty();
            virtual void setThreadsCount(uintt threadsCount);
        };

        template<typename T> class ThreadsCPU : public ThreadsCountProperty {
            uintt* m_bmap;
        protected:
            ThreadData<T>* m_threadData;
            uintt* getBMap() const;
        public:
            ThreadsCPU();
            virtual ~ThreadsCPU();
            void setThreadsCount(uintt threadsCount);
        };

        class AdditionOperation : public math::IAdditionOperation,
        public ThreadsCPU<AdditionOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            AdditionOperation();
            ~AdditionOperation();
        };

        class SubstracionOperation : public math::ISubstracionOperation,
        public ThreadsCPU<SubstracionOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            SubstracionOperation();
            ~SubstracionOperation();
        };

        class DotProductOperation : public math::IDotProductOperation,
        public ThreadsCPU<DotProductOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            DotProductOperation();
            ~DotProductOperation();
        };

        class MultiplicationConstOperation :
        public math::IMultiplicationConstOperation,
        public ThreadsCPU<MultiplicationConstOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            MultiplicationConstOperation();
            ~MultiplicationConstOperation();
        };

        class ExpOperation : public math::IExpOperation,
        public ThreadsCountProperty {
            uintt serieLimit;
        protected:
            DotProductOperation dotProduct;
            MultiplicationConstOperation multiplication;
            AdditionOperation addition;
        public:
            void execute();
            ExpOperation();
            ~ExpOperation();
        };

        class DiagonalizationOperation :
        public math::IDiagonalizationOperation,
        public ThreadsCPU<DiagonalizationOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            DiagonalizationOperation();
            virtual ~DiagonalizationOperation();
        };

        class TensorProductOperation :
        public math::ITensorProductOperation,
        public ThreadsCPU<TensorProductOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            TensorProductOperation();
            ~TensorProductOperation();
        };

        class MagnitudeOperation :
        public math::IMagnitudeOperation,
        public ThreadsCPU<MagnitudeOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            MagnitudeOperation();
            ~MagnitudeOperation();
        };

        class TransposeOperation :
        public math::ITransposeOperation,
        public ThreadsCPU<TransposeOperation> {
            static void Execute(void* ptr);
        protected:
            void execute();
        public:
            TransposeOperation();
            ~TransposeOperation();
        };

        class QRDecomposition :
        public math::IQRDecomposition,
        public ThreadsCountProperty {
            static void Execute(void* ptr);
            math::Matrix* R1;
            math::Matrix* Q1;
            math::Matrix* G;
            math::Matrix* GT;
            DotProductOperation dotProduct;
            TransposeOperation transpose;
            inline void prepareGMatrix(math::Matrix* A,
                    uintt column, uintt row,
                    math::Matrix* G);
        protected:
            void execute();
        public:
            QRDecomposition();
            ~QRDecomposition();
        };

        class DeterminantOperation :
        public math::IDeterminantOperation,
        public ThreadsCountProperty {
            QRDecomposition m_qrDecomposition;
            math::Matrix* m_q;
            math::Matrix* m_r;
        protected:
            math::Status beforeExecution();
            void execute();
        public:
            DeterminantOperation();
            ~DeterminantOperation();
        };

        class MathOperations;

        /**
         * Implementation of implicity reverse arnoldi method.
         * 
         */
        class IraMethod : public math::IIraMethod,
        public ThreadsCountProperty {
            MatrixUtils* mu;
            MatrixAllocator* ma;
            MatrixCopier* mc;
            floatt diff;
            math::Matrix* w;
            math::Matrix* f;
            math::Matrix* f1;
            math::Matrix* vh;
            math::Matrix* h;
            math::Matrix* s;
            math::Matrix* vs;
            math::Matrix* V;
            math::Matrix* EV;
            math::Matrix* EV1;
            math::Matrix* EQ1;
            math::Matrix* EQ2;
            math::Matrix* EQ3;
            math::Matrix* V1;
            math::Matrix* V2;
            math::Matrix* transposeV;
            math::Matrix* H;
            math::Matrix* HC;
            math::Matrix* H1;
            math::Matrix* H2;
            math::Matrix* A2;
            math::Matrix* I;
            math::Matrix* A;
            math::Matrix* A1;
            math::Matrix* v;
            math::Matrix* q;
            math::Matrix* QT;
            math::Matrix* Q1;
            math::Matrix* Q2;
            math::Matrix* R1;
            math::Matrix* R2;
            math::Matrix* HO;
            math::Matrix* HO1;
            math::Matrix* Q;
            math::Matrix* QJ;
            std::vector<Complex> notSorted;
            std::vector<Complex> wanted;
            std::vector<uintt> wantedIndecies;
            std::vector<Complex> unwanted;
            uintt m_k;
            floatt m_rho;
            uintt m_wantedCount;
            MathOperations& m_mathOperations;
            floatt getReDiagonal(math::Matrix* matrix, intt index);
            floatt getImDiagonal(math::Matrix* matrix, intt index);
            bool isEigenValue(math::Matrix* matrix, intt index);
            bool continueProcedure();
            bool executeArnoldiFactorization(bool init = true, intt initj = 0);
            void calculateH(int i);
            math::Matrix* oldA;
            void alloc(math::Matrix* A);
            void dealloc();
            floatt getLargestDiagonal(math::Matrix* H) const;
            floatt getSmallestDiagonal(math::Matrix* H) const;

        protected:
            math::Matrix* getV() const;
            math::Matrix* getW() const;
            math::Matrix* getA() const;
        public:
            IraMethod(MathOperations* mathOperations);
            IraMethod(MatrixModule* matrixModule,
                    MatrixStructureUtils* matrixStructureUtils,
                    MathOperations* mathOperations);
            virtual ~IraMethod();
            void setHSize(uintt k);
            void setRho(floatt rho);
            virtual void multiply(math::Matrix* a, math::Matrix* b,
                    math::Matrix* c, bool first);
            void execute();
        };

        class IraMethodCallback : public IraMethod,
        public utils::CallbacksManager {
            floatt* m_reoutputs;
            floatt* m_reoutputs1;
            floatt* m_imoutputs;
            floatt* m_imoutputs1;
            uintt* m_matrixEntries;
            uintt m_realCount;
            uintt m_count;
            uintt m_count1;
            uintt m_count2;
            uintt m_rows;
            uintt m_rows1;
            uintt m_index;

            struct Data {
                utils::Thread thread;
                uintt m_brow;
                uintt m_erow;
                floatt* m_reoutputs;
                floatt* m_imoutputs;
                floatt* m_reoutputs1;
                floatt* m_imoutputs1;
                uintt m_count2;
                uintt m_beginIndex;
                math::Matrix* w;

                void calculate(uintt index, uintt count,
                        uintt begin, uintt size, uint count2);
            };
            static void ThreadFunction(void* ptr);
            std::vector<Data*> m_threads;
        public:
            Status beforeExecution();
            Status afterExecution();

            static const int EVENT_MATRIX_MULTIPLICATION;

            class Event {
                floatt* m_reoutputs;
                floatt* m_imoutputs;
                uintt* m_matrixEntries;
                uintt m_count;
                Event(floatt* reoutputs, floatt* imoutpus,
                        uintt* matrixEntries, uintt count);
                Event();
                virtual ~Event();
                void setPointers(floatt* reoutputs, floatt* imoutpus,
                        uintt* matrixEntries);
                void setCount(uintt count);
                friend class IraMethodCallback;
            public:
                uintt getCount() const;
                floatt* getReOutputs() const;
                floatt* getImOutputs() const;
                uintt* getMatrixEntries() const;

            };

            IraMethodCallback(MathOperations* mathOperations, uintt realCount);
            virtual ~IraMethodCallback();
            bool isFinish;
            void preMultiply(math::Matrix* a, math::Matrix* b,
                    math::Matrix* c);
            void postMultiply(math::Matrix* a, math::Matrix* b,
                    math::Matrix* c);
            void multiply(math::Matrix* a, math::Matrix* b,
                    math::Matrix* c, bool first);
        private:
            Event* m_event;
        };

        class MathOperations : public utils::Module {
#ifdef DEBUG
            std::map<void*, std::string> valuesNames;
#endif
            int threadsCount;
            int serieLimit;
            AdditionOperation additionOperation;
            SubstracionOperation substracionOperation;
            DotProductOperation dotProductOperation;
            TensorProductOperation tensorProductOperation;
            DiagonalizationOperation diagonalizationOperation;
            ExpOperation expOperation;
            MultiplicationConstOperation multiplicationConstOperation;
            MagnitudeOperation magnitudeOperation;
            TransposeOperation m_transposeOperation;
            DeterminantOperation m_determinantOperation;
            QRDecomposition m_qrDecomposition;
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
            MathOperations();
            virtual ~MathOperations();
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

        template<typename T> ThreadsCPU<T>::ThreadsCPU() : ThreadsCountProperty(),
        m_bmap(NULL), m_threadData(NULL) {
            m_threadData = new ThreadData<T>[m_threadsCount];
            m_bmap = utils::mapper::allocMap(m_threadsCount);
        }

        template<typename T> ThreadsCPU<T>::~ThreadsCPU() {
            if (m_bmap) {
                utils::mapper::freeMap(m_bmap);
                delete[] m_threadData;
            }
        }

        template<typename T> uintt* ThreadsCPU<T>::getBMap() const {
            return this->m_bmap;
        }

        template<typename T> void ThreadsCPU<T>::setThreadsCount(uintt threadsCount) {
            if (this->m_threadsCount < threadsCount || m_bmap == NULL) {
                if (m_bmap) {
                    utils::mapper::freeMap(m_bmap);
                    delete[] m_threadData;
                    m_bmap = NULL;
                }
                if (m_bmap == NULL) {
                    m_threadData = new ThreadData<T>[threadsCount];
                    m_bmap = utils::mapper::allocMap(threadsCount);
                }
            }
            this->m_threadsCount = threadsCount;
        }
    }
}
#endif	/* MATRIXOPERATIONSCPU_H */

