#include <math.h>
#include "ArnoldiMethodDeviceImpl.h"
#include "DeviceMatrixStructure.h"
#include "ArnoldiProcedures.h"

#ifdef DEBUG
#define PRINT_STATUS(d) if(d!=0) { fprintf(stderr,"Status == %d \n",d); abort();}
#else
#define PRINT_STATUS(d) d
#endif

const char* kernelsFiles[] = {
    "/home/mmatula/Ogla/ArnoldiPackage/dist/Debug/albert/libArnoldiPackage.cubin",
    NULL
};

#define MIN_VALUE 0.001


namespace math {

    ArnoldiMethodGpu::ArnoldiMethodGpu(MathOperationsCuda* mathOperations) :
    IArnoldiMethod(DeviceMatrixModules::GetInstance(),
    DeviceMatrixStructureUtils::GetInstance()),
    m_operations(mathOperations) {
        this->m_rho = 1. / 3.14;
        this->m_k = 0;
        this->m_wantedCount = 0;
        oldA = NULL;
        m_image = NULL;
    }

    ArnoldiMethodGpu::ArnoldiMethodGpu(
            MatrixModule* matrixModule,
            MatrixStructureUtils* matrixStructureUtils,
            MathOperationsCuda* mathOperations) :
    IArnoldiMethod(matrixModule, matrixStructureUtils),
    m_operations(mathOperations) {
        this->m_rho = 1. / 3.14;
        this->m_k = 0;
        this->m_wantedCount = 0;
        diff = -10.552;
        m_image = NULL;
    }

    math::Matrix* ArnoldiMethodGpu::getV() const {
        return this->v.m_matrix;
    }

    math::Matrix* ArnoldiMethodGpu::getW() const {
        return this->w.m_matrix;
    }

    math::Matrix* ArnoldiMethodGpu::getA() const {
        return this->A.m_matrix;
    }

    void ArnoldiMethodGpu::setHSize(uintt k) {
        this->m_k = k;
    }

    void ArnoldiMethodGpu::setRho(floatt rho) {
        this->m_rho = rho;
    }

    ArnoldiMethodGpu::~ArnoldiMethodGpu() {
        dealloc();
        ::cuda::Kernel::FreeImage(m_image);
    }

    floatt ArnoldiMethodGpu::getReDiagonal(math::Matrix* matrix, intt index) {
        if (matrix->reValues == NULL) {
            return 0;
        }
        return matrix->reValues[index + matrix->columns * index];
    }

    floatt ArnoldiMethodGpu::getImDiagonal(math::Matrix* matrix, intt index) {
        if (matrix->imValues == NULL) {
            return 0;
        }
        return matrix->imValues[index + matrix->columns * index];
    }

    bool ArnoldiMethodGpu::isEigenValue(math::Matrix* matrix, intt index) {
        floatt v = matrix->reValues[(index - 1) + matrix->columns * index];
        if (fabs(v) < MATH_VALUE_LIMIT) {
            return true;
        }
        return false;
    }

    ArnoldiMethodGpu::MatrixData::MatrixData() :
    m_matrix(NULL),
    m_matrixStructure(NULL) {
    }

    MatrixStructure** ArnoldiMethodGpu::MatrixData::getMatrixStructurePointer() {
        debug("getMatrixStructure = %p", m_matrixStructure);
        return &m_matrixStructure;
    }

    void ArnoldiMethodGpu::MatrixData::alloc(math::Matrix* A, intt columns,
            intt rows, ArnoldiMethodGpu* thiz) {
        m_matrix = cuda::NewDeviceMatrix(A);
        m_matrixStructure = thiz->m_matrixStructureUtils->newMatrixStructure();
        thiz->m_matrixStructureUtils->setMatrix(m_matrixStructure, m_matrix);
    }

    void ArnoldiMethodGpu::MatrixData::dealloc(ArnoldiMethodGpu* thiz) {
        thiz->m_module->deleteMatrix(m_matrix);
        thiz->m_matrixStructureUtils->deleteMatrixStructure(m_matrixStructure);
    }

    void ArnoldiMethodGpu::alloc(math::Matrix* A) {
        if (oldA == NULL || (A->rows != oldA->rows && A->columns != oldA->columns)) {
            dealloc();
            debugAssert(m_k != 0);
            w.alloc(A, 1, A->rows, this);
            v.alloc(A, 1, A->rows, this);
            f.alloc(A, 1, A->rows, this);
            f1.alloc(A, 1, A->rows, this);
            vh.alloc(A, 1, A->rows, this);
            h.alloc(A, 1, m_k, this);
            s.alloc(A, 1, m_k, this);
            vs.alloc(A, 1, A->rows, this);
            V.alloc(A, m_k, A->rows, this);
            V1.alloc(A, m_k, A->rows, this);
            V2.alloc(A, m_k, A->rows, this);
            EQ1.alloc(A, 1, A->rows, this);
            EQ2.alloc(A, 1, A->rows, this);
            EQ3.alloc(A, 1, A->rows, this);
            EV.alloc(A, m_k, A->rows, this);
            EV1.alloc(A, m_k, A->rows, this);
            H.alloc(A, m_k, m_k, this);
            HO.alloc(A, m_k, m_k, this);
            H1.alloc(A, m_k, m_k, this);
            Q.alloc(A, m_k, m_k, this);
            Q1.alloc(A, m_k, m_k, this);
            Q2.alloc(A, m_k, m_k, this);
            QT.alloc(A, m_k, m_k, this);
            R1.alloc(A, m_k, m_k, this);
            R2.alloc(A, m_k, m_k, this);
            QJ.alloc(A, m_k, m_k, this);
            I.alloc(A, m_k, m_k, this);
            G.alloc(A, m_k, m_k, this);
            GT.alloc(A, m_k, m_k, this);
            A1.alloc(A, A->columns, A->columns, this);
            transposeV.alloc(A, A->rows, m_k, this);
            oldA = A;
        }
    }

    void ArnoldiMethodGpu::dealloc() {
        if (NULL != oldA) {
            debugFunc();
            (w).dealloc(this);
            (QJ).dealloc(this);
            (Q).dealloc(this);
            (f).dealloc(this);
            (vh).dealloc(this);
            (h).dealloc(this);
            (s).dealloc(this);
            (vs).dealloc(this);
            (V).dealloc(this);
            (H).dealloc(this);
            (H1).dealloc(this);
            (I).dealloc(this);
            (v).dealloc(this);
            (transposeV).dealloc(this);
            (A1).dealloc(this);
            (V1).dealloc(this);
            (V2).dealloc(this);
            (EV).dealloc(this);
            (EV1).dealloc(this);
            (R1).dealloc(this);
            (R2).dealloc(this);
            (QT).dealloc(this);
            (Q1).dealloc(this);
            (HO).dealloc(this);
            (EQ1).dealloc(this);
            (EQ2).dealloc(this);
            (Q2).dealloc(this);
            (EQ3).dealloc(this);
            (G).dealloc(this);
            (GT).dealloc(this);
            debugFunc();
        }
    }

    void ArnoldiMethodGpu::execute() {
        debugFunc();
        m_image = ::cuda::Kernel::LoadImage(kernelsFiles);
        debugFunc();
        alloc(m_matrix);
        debugFunc();

        floatt a[] = {209876.114322, 5543.454862, -1.931923, 150.393653,
            5545.494910, 204558.192376, 5615.605322, 152.459566,
            0.000000, 5615.551427, 209721.857008, 220.557474,
            0.000000, 0.000000, 72.205295, 204259.343999};

        cuda::CopyHostArraysToDeviceMatrix(H1.m_matrix, a, NULL);

        void* params[] = {
            H1.getMatrixStructurePointer(),
            Q.getMatrixStructurePointer(),
            R1.getMatrixStructurePointer(),
            Q1.getMatrixStructurePointer(),
            QJ.getMatrixStructurePointer(),
            Q2.getMatrixStructurePointer(),
            R2.getMatrixStructurePointer(),
            G.getMatrixStructurePointer(),
            GT.getMatrixStructurePointer()
        };
        debugFunc();
        m_kernel.setThreadsCount(m_k, m_k);
        DeviceMatrixModules::GetInstance()->getMatrixPrinter()->printReMatrix("H1 = ", H1.m_matrix);
        CUDAKernel_CalculateHQ(params, m_kernel, m_image);
        DeviceMatrixModules::GetInstance()->getMatrixPrinter()->printReMatrix("H1 = ", H1.m_matrix);
        //cuda::Kernel::ExecuteKernel("CUDAKernel_Execute", params1, m_kernel, m_image);
        debugFunc();
        dealloc();
    }

    void ArnoldiMethodGpu::multiply(math::Matrix* a, math::Matrix* b,
            math::Matrix* c, bool first) {
        PRINT_STATUS(m_operations->dotProduct(a, b, c));
    }

    math::Status ArnoldiMethodCallbackGpu::beforeExecution() {
        return ArnoldiMethodGpu::beforeExecution();
    }

    math::Status ArnoldiMethodCallbackGpu::afterExecution() {
        return ArnoldiMethodGpu::afterExecution();
    }

    floatt ArnoldiMethodGpu::getLargestDiagonal(math::Matrix* H) const {
        floatt output = H->reValues[0];
        for (uintt fa = 1; fa < H->columns; ++fa) {
            floatt v = H->reValues[fa * H->columns + fa];
            if (v > output) {
                output = v;
            }
        }
        return output;
    }

    floatt ArnoldiMethodGpu::getSmallestDiagonal(math::Matrix* H) const {
        floatt output = H->reValues[0];
        for (uintt fa = 1; fa < H->columns; ++fa) {
            floatt v = H->reValues[fa * H->columns + fa];
            if (v < output) {
                output = v;
            }
        }
        return output;
    }

    ArnoldiMethodCallbackGpu::ArnoldiMethodCallbackGpu(MathOperationsCuda* mathOperations,
            uintt realCount) :
    ArnoldiMethodGpu(mathOperations) {
        m_event = new Event();
        m_realCount = realCount;
        m_reoutputs = new floatt[realCount];
        m_reoutputs1 = new floatt[realCount];
        m_imoutputs = new floatt[realCount];
        m_imoutputs1 = new floatt[realCount];
        m_matrixEntries = new uintt[realCount * 2];
        m_count = 0;
        m_index = 0;
    }

    ArnoldiMethodCallbackGpu::~ArnoldiMethodCallbackGpu() {
        delete m_event;
        if (m_reoutputs) {
            delete[] m_reoutputs;
        }
        if (m_imoutputs) {
            delete[] m_imoutputs;
        }
        if (m_reoutputs1) {
            delete[] m_reoutputs1;
        }
        if (m_imoutputs1) {
            delete[] m_imoutputs1;
        }
        if (m_matrixEntries) {
            delete[] m_matrixEntries;
        }
    }

    void ArnoldiMethodCallbackGpu::preMultiply(math::Matrix* a, math::Matrix* b,
            math::Matrix* c) {
        math::Matrix* v = c;
        m_count = 0;
        isFinish = false;
        m_index = 0;
        for (uintt fa1 = m_rows; fa1 < v->rows; ++fa1) {
            for (uintt fa = 0; fa < v->rows; ++fa) {
                floatt re = 0;
                if (v->reValues != 0) {
                    re = v->reValues[fa];
                }
                floatt im = 0;
                if (v->imValues != 0) {
                    im = v->imValues[fa];
                }
                if (re != 0 || im != 0) {
                    m_reoutputs[m_index] = re;
                    m_imoutputs[m_index] = im;
                    m_matrixEntries[m_index * 2] = fa;
                    m_matrixEntries[m_index * 2 + 1] = fa1;
                    m_count++;
                    m_index++;
                    if (m_count >= m_realCount) {
                        m_count = m_count1;
                        m_rows1 = fa1;
                        return;
                    }
                }
            }
            if (fa1 == 0) {
                m_count2 = m_count;
            }
            m_count1 = m_count;
        }
        m_rows1 = v->rows;
        isFinish = true;
    }

    void ArnoldiMethodCallbackGpu::Data::calculate(uintt index, uintt count,
            uintt begin, uintt size, uint count2) {
        uintt diff = size / count;
        uintt dindex = diff * index;
        if (index == count - 1) {
            m_brow = begin + dindex;
            m_erow = begin + size;
        } else {
            m_brow = begin + dindex;
            m_erow = begin + diff * (index + 1);
        }
        m_beginIndex = dindex*count2;
        this->m_count2 = count2;
    }

    void ArnoldiMethodCallbackGpu::postMultiply(math::Matrix* a, math::Matrix* b,
            math::Matrix* c) {

    }

    void ArnoldiMethodCallbackGpu::multiply(math::Matrix* a, math::Matrix* b,
            math::Matrix* c, bool first) {
        if (first == false) {
            m_event->setPointers(m_reoutputs1, m_imoutputs1,
                    m_matrixEntries);
            m_index = 0;
            m_count1 = 0;
            m_rows1 = 0;
            m_rows = 0;
            isFinish = false;
            while (isFinish == false) {
                preMultiply(a, b, c);
                m_event->setCount(m_count);
                //invokeCallbacks(EVENT_MATRIX_MULTIPLICATION, m_event);
                postMultiply(a, b, c);
            }
        } else {
            host::CopyMatrix(a, b);
        }
    }

    const int ArnoldiMethodCallbackGpu::EVENT_MATRIX_MULTIPLICATION = 0;

    ArnoldiMethodCallbackGpu::Event::Event(floatt* reoutputs,
            floatt* imoutputs,
            uintt* matrixEntries, uintt count) {
        m_reoutputs = reoutputs;
        m_imoutputs = imoutputs;
        m_matrixEntries = matrixEntries;
        m_count = count;
    }

    ArnoldiMethodCallbackGpu::Event::Event() {
        m_reoutputs = NULL;
        m_imoutputs = NULL;
        m_matrixEntries = NULL;
        m_count = 0;
    }

    ArnoldiMethodCallbackGpu::Event::~Event() {
    }

    void ArnoldiMethodCallbackGpu::Event::setPointers(floatt* reoutputs,
            floatt* imoutputs,
            uintt* matrixEntries) {
        m_reoutputs = reoutputs;
        m_imoutputs = imoutputs;
        m_matrixEntries = matrixEntries;
    }

    void ArnoldiMethodCallbackGpu::Event::setCount(uintt count) {
        m_count = count;
        memset(m_reoutputs, 0, count * sizeof (floatt));
        memset(m_imoutputs, 0, count * sizeof (floatt));
    }

    uintt ArnoldiMethodCallbackGpu::Event::getCount() const {
        return m_count;
    }

    floatt* ArnoldiMethodCallbackGpu::Event::getReOutputs() const {
        return m_reoutputs;
    }

    floatt* ArnoldiMethodCallbackGpu::Event::getImOutputs() const {
        return m_imoutputs;
    }

    uintt* ArnoldiMethodCallbackGpu::Event::getMatrixEntries() const {
        return m_matrixEntries;
    }
}
