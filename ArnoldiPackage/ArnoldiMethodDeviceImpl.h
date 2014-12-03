#ifndef CUDA_ARNOLDIMETHODIMPL_H
#define	CUDA_ARNOLDIMETHODIMPL_H

#include <vector>
#include "IArnoldiMethod.h"
#include "Math.h"
#include "MathOperationsCuda.h"

/**
 * Implementation of implicity reverse arnoldi method.
 * 
 */
namespace math {

    class ArnoldiMethodGpu :
    public math::IArnoldiMethod {
        void* m_image;
        ::cuda::Kernel m_kernel;
        floatt diff;

        class MatrixData {
            MatrixStructure* m_matrixStructure;
        public:
            math::Matrix* m_matrix;
            MatrixStructure** getMatrixStructurePointer();
            MatrixData();
            void alloc(math::Matrix* A, intt columns, intt rows, ArnoldiMethodGpu* thiz);
            void dealloc(ArnoldiMethodGpu* thiz);
        };
        friend class MatrixData;
        MatrixData w;
        MatrixData f;
        MatrixData f1;
        MatrixData vh;
        MatrixData h;
        MatrixData s;
        MatrixData vs;
        MatrixData V;
        MatrixData EV;
        MatrixData EV1;
        MatrixData EQ1;
        MatrixData EQ2;
        MatrixData EQ3;
        MatrixData V1;
        MatrixData V2;
        MatrixData transposeV;
        MatrixData H;
        MatrixData HC;
        MatrixData H1;
        MatrixData H2;
        MatrixData A2;
        MatrixData I;
        MatrixData A;
        MatrixData A1;
        MatrixData v;
        MatrixData q;
        MatrixData QT;
        MatrixData Q1;
        MatrixData Q2;
        MatrixData R1;
        MatrixData R2;
        MatrixData HO;
        MatrixData HO1;
        MatrixData Q;
        MatrixData QJ;
        MatrixData G;
        MatrixData GT;
        typedef std::vector<Complex> Complecies;
        typedef std::vector<uintt> Indecies;
        Complecies notSorted;
        Complecies wanted;
        Complecies unwanted;
        Indecies wantedIndecies;

        uintt m_k;
        floatt m_rho;
        uintt m_wantedCount;
        math::Matrix* m_deviceMatrix;

        math::MathOperationsCuda* m_operations;
        floatt getReDiagonal(math::Matrix* matrix, intt index);
        floatt getImDiagonal(math::Matrix* matrix, intt index);
        bool isEigenValue(math::Matrix* matrix, intt index);
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
        ArnoldiMethodGpu(math::MathOperationsCuda* mathOperations);
        ArnoldiMethodGpu(MatrixModule* matrixModule,
                MatrixStructureUtils* matrixStructureUtils,
                math::MathOperationsCuda* mathOperations);
        virtual ~ArnoldiMethodGpu();
        void setHSize(uintt k);
        void setRho(floatt rho);
        virtual void multiply(math::Matrix* a, math::Matrix* b,
                math::Matrix* c, bool first);
        void execute();
    };

    class ArnoldiMethodCallbackGpu :
    public ArnoldiMethodGpu,
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
            friend class ArnoldiMethodCallbackGpu;
        public:
            uintt getCount() const;
            floatt* getReOutputs() const;
            floatt* getImOutputs() const;
            uintt* getMatrixEntries() const;
        };

        ArnoldiMethodCallbackGpu(MathOperationsCuda* mathOperations,
                uintt realCount);
        virtual ~ArnoldiMethodCallbackGpu();
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

