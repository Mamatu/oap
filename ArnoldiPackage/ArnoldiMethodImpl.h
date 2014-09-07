#ifndef CPU_ARNOLDIMETHODINTERNAL_H
#define	CPU_ARNOLDIMETHODINTERNAL_H

#include <vector>
#include "ThreadsCpu.h"
#include "IArnoldiMethod.h"
#include "MathOperationsCpu.h"

/**
 * Implementation of implicity reverse arnoldi method.
 * 
 */
namespace math {

    class ArnoldiMethodCpu :
    public math::IArnoldiMethod,
    public ThreadsCountProperty {
        MatrixUtils* m_utils;
        MatrixAllocator* m_allocator;
        MatrixCopier* m_copier;
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
        math::MathOperationsCpu* m_operations;
        floatt getReDiagonal(math::Matrix* matrix, intt index);
        floatt getImDiagonal(math::Matrix* matrix, intt index);
        bool isEigenValue(math::Matrix* matrix, intt index);
        bool continueProcedure();
        bool testProcedure(uintt fa);
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
        ArnoldiMethodCpu(math::MathOperationsCpu* mathOperations);
        ArnoldiMethodCpu(MatrixModule* matrixModule,
                MatrixStructureUtils* matrixStructureUtils,
                math::MathOperationsCpu* mathOperations);
        virtual ~ArnoldiMethodCpu();
        void setHSize(uintt k);
        void setRho(floatt rho);
        virtual void multiply(math::Matrix* a, math::Matrix* b,
                math::Matrix* c, bool first);
        void execute();
    };

    class ArnoldiMethodCallbackCpu :
    public ArnoldiMethodCpu,
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
        math::Status beforeExecution();
        math::Status afterExecution();

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
            friend class ArnoldiMethodCallbackCpu;
        public:
            uintt getCount() const;
            floatt* getReOutputs() const;
            floatt* getImOutputs() const;
            uintt* getMatrixEntries() const;
        };

        ArnoldiMethodCallbackCpu(MathOperationsCpu* mathOperations,
                uintt realCount);
        virtual ~ArnoldiMethodCallbackCpu();
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
}


#endif	/* ARNOLDIMETHOD_H */

