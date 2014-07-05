#include <math.h>
#include "MathOperationsCpu.h"
#include "HostMatrixStructure.h"

#ifdef DEBUG
#define PS(d) if(d!=0) { fprintf(stderr,"Status == %d \n",d); abort();}
#else
#define PS(d) d
#endif

namespace math {
    namespace cpu {

        bool IsTriangular(math::Matrix* matrix, uintt count) {
            uintt index = 0;
            for (uintt fa = 0; fa < matrix->columns - 1; ++fa) {
                floatt revalue = matrix->reValues[fa + matrix->columns * (fa + 1)];
                floatt imvalue = 0;
                if (matrix->imValues) {
                    imvalue = matrix->imValues[fa + matrix->columns * (fa + 1)];
                }
                if ((-0.00000000001 < revalue && revalue < 0.00000000001) &&
                        (-0.00000000001 < imvalue && imvalue < 0.00000000001)) {
                    index++;
                }
            }
            if (index >= count) {
                return true;
            } else {
                return false;
            }
        }

        inline void switchPointer(math::Matrix*& a, math::Matrix*& b) {
            math::Matrix* temp = b;
            b = a;
            a = temp;
        }

        IraMethod::IraMethod(MatrixModule* matrixModule,
                MatrixStructureUtils* matrixStructureUtils,
                MathOperations* mathOperations) :
        IIraMethod(matrixModule, matrixStructureUtils),
        m_mathOperations(*mathOperations) {
            this->m_rho = 1. / 3.14;
            this->m_k = 0;
            this->m_wantedCount = 0;
        }

        math::Matrix* IraMethod::getV() const {
            return this->v;
        }

        math::Matrix* IraMethod::getW() const {
            return this->w;
        }

        math::Matrix* IraMethod::getA() const {
            return this->A;
        }

        IraMethod::IraMethod(MathOperations* mathOperations) :
        IIraMethod(&HostMatrixModules::GetInstance(),
        HostMatrixStructureUtils::GetInstance(&HostMatrixModules::GetInstance())),
        m_mathOperations(*mathOperations) {
            this->m_rho = 1. / 3.14;
            this->m_k = 0;
            this->m_wantedCount = 0;
            oldA = NULL;
        }

        void IraMethod::setHSize(uintt k) {
            this->m_k = k;
        }

        void IraMethod::setRho(floatt rho) {
            this->m_rho = rho;
        }

        IraMethod::~IraMethod() {
            dealloc();
        }

        bool IraMethod::continueProcedure() {
            int st = 0;
            m_mathOperations.multiply(EV, V, Q);
            for (uintt fa = 0; fa < wanted.size(); ++fa) {
                uintt index = wantedIndecies[fa];
                Complex b = notSorted[index];
                mc->getVector(v, v->rows, EV, index);
                multiply(EQ1, A, v, false);
                m_mathOperations.multiply(EQ2, EV, &(b.re), &(b.im));
                fprintf(stderr, "wanted == %f %f\n", b.re, b.im);
                host::PrintMatrix("EQ1 = ", EQ1);
                host::PrintMatrix("EQ2 = ", EQ2);
            }
            for (int fa = 0; fa < unwanted.size(); ++fa) {
                fprintf(stderr, "unwanted == %f %f\n", unwanted[fa].re, unwanted[fa].im);
            }
            abort();
            return true;
        }

        floatt IraMethod::getReDiagonal(math::Matrix* matrix, intt index) {
            if (matrix->reValues == NULL) {
                return 0;
            }
            return matrix->reValues[index + matrix->columns * index];
        }

        floatt IraMethod::getImDiagonal(math::Matrix* matrix, intt index) {
            if (matrix->imValues == NULL) {
                return 0;
            }
            return matrix->imValues[index + matrix->columns * index];
        }

        bool IraMethod::isEigenValue(math::Matrix* matrix, intt index) {
            floatt v = matrix->reValues[(index - 1) + matrix->columns * index];
            if (fabs(v) < MATH_VALUE_LIMIT) {
                return true;
            }
            return false;
        }

        void IraMethod::alloc(math::Matrix* A) {
            if (oldA == NULL ||
                    (A->rows != oldA->rows && A->columns != oldA->columns)) {
                dealloc();
                debugFunc();
                w = m_matrixModule->newMatrix(A, 1, A->rows);
                v = m_matrixModule->newMatrix(A, 1, A->rows);
                f = m_matrixModule->newMatrix(A, 1, A->rows);
                f1 = m_matrixModule->newMatrix(A, 1, A->rows);
                vh = m_matrixModule->newMatrix(A, 1, A->rows);
                h = m_matrixModule->newMatrix(A, 1, m_k);
                s = m_matrixModule->newMatrix(A, 1, m_k);
                vs = m_matrixModule->newMatrix(A, 1, A->rows);
                V = m_matrixModule->newMatrix(A, m_k, A->rows);
                V1 = m_matrixModule->newMatrix(A, m_k, A->rows);
                V2 = m_matrixModule->newMatrix(A, m_k, A->rows);
                EQ1 = m_matrixModule->newMatrix(A, 1, A->rows);
                EQ2 = m_matrixModule->newMatrix(A, 1, A->rows);
                EV = m_matrixModule->newMatrix(A, m_k, A->rows);
                H = m_matrixModule->newMatrix(A, m_k, m_k);
                HO = m_matrixModule->newMatrix(A, m_k, m_k);
                H1 = m_matrixModule->newMatrix(A, m_k, m_k);
                Q = m_matrixModule->newMatrix(A, m_k, m_k);
                Q1 = m_matrixModule->newMatrix(A, m_k, m_k);
                QT = m_matrixModule->newMatrix(A, m_k, m_k);
                R1 = m_matrixModule->newMatrix(A, m_k, m_k);
                R2 = m_matrixModule->newMatrix(A, m_k, m_k);
                QJ = m_matrixModule->newMatrix(A, m_k, m_k);
                I = m_matrixModule->newMatrix(A, m_k, m_k);
                A1 = m_matrixModule->newMatrix(A, A->columns, A->columns);
                transposeV = m_matrixModule->newMatrix(A, A->rows, m_k);
                oldA = A;
                debugFunc();
            }
        }

        void IraMethod::dealloc() {
            if (oldA) {
                debugFunc();
                m_matrixModule->deleteMatrix(w);
                m_matrixModule->deleteMatrix(QJ);
                m_matrixModule->deleteMatrix(Q);
                m_matrixModule->deleteMatrix(f);
                m_matrixModule->deleteMatrix(vh);
                m_matrixModule->deleteMatrix(h);
                m_matrixModule->deleteMatrix(s);
                m_matrixModule->deleteMatrix(vs);
                m_matrixModule->deleteMatrix(V);
                m_matrixModule->deleteMatrix(H);
                m_matrixModule->deleteMatrix(H1);
                m_matrixModule->deleteMatrix(I);
                m_matrixModule->deleteMatrix(v);
                m_matrixModule->deleteMatrix(transposeV);
                m_matrixModule->deleteMatrix(A1);
                m_matrixModule->deleteMatrix(V1);
                m_matrixModule->deleteMatrix(V2);
                m_matrixModule->deleteMatrix(EV);
                m_matrixModule->deleteMatrix(R1);
                m_matrixModule->deleteMatrix(R2);
                m_matrixModule->deleteMatrix(QT);
                m_matrixModule->deleteMatrix(Q1);
                m_matrixModule->deleteMatrix(HO);
                debugFunc();
            }
        }

        void IraMethod::execute() {
            mu = m_matrixModule->getMatrixUtils();
            ma = m_matrixModule->getMatrixAllocator();
            mc = m_matrixModule->getMatrixCopier();

            m_wantedCount = m_count;
            math::Matrix* A = m_matrix;
            //m_mathOperations.setThreadsCount(m_threadsCount);
            alloc(A);
            v->reValues[0] = 1;
            floatt tempLenght = 0;
            m_mathOperations.magnitude(&tempLenght, v);
            tempLenght = 1. / tempLenght;
            m_mathOperations.multiply(v, v, &tempLenght);
            mc->setVector(V, 0, v, v->rows);
            this->A = A;
            bool finish = false;
            this->executeArnoldiFactorization();
            for (intt fax = 0; finish == false; ++fax) {
                unwanted.clear();
                wanted.clear();
                wantedIndecies.clear();
                this->calculateH(H->columns - m_wantedCount);
                if (continueProcedure() == true) {
                    mu->setIdentityMatrix(Q);
                    mu->setIdentityMatrix(QJ);
                    int p = unwanted.size();
                    int k = wanted.size();
                    for (intt fa = 0; fa < p; ++fa) {
                        mu->setDiagonalMatrix(I, unwanted[fa].re, unwanted[fa].im);
                        fprintf(stderr, "%f %f \n", unwanted[fa].re, unwanted[fa].im);
                        PS(m_mathOperations.substract(I, H, I));
                        PS(m_mathOperations.qrDecomposition(Q1, R1, I));
                        PS(m_mathOperations.transpose(QT, Q1));
                        PS(m_mathOperations.dotProduct(HO, H, Q1));
                        PS(m_mathOperations.dotProduct(H, QT, HO));
                        PS(m_mathOperations.dotProduct(Q, QJ, Q1));
                        if (fa < p - 1) {
                            switchPointer(Q, QJ);
                        }
                    }
                    floatt reqm_k = Q->reValues[Q->columns * (Q->rows - 1) + k];
                    floatt imqm_k = 0;
                    if (Q->imValues) {
                        imqm_k = Q->imValues[Q->columns * (Q->rows - 1) + k];
                    }
                    floatt reBm_k = H->reValues[H->columns * (k + 1) + k];
                    floatt imBm_k = 0;
                    if (H->imValues) {
                        imBm_k = H->imValues[H->columns * (k + 1) + k];
                    }
                    mc->getVector(v, v->rows, V, k);
                    PS(m_mathOperations.multiply(f1, v, &reBm_k, &imBm_k));
                    PS(m_mathOperations.multiply(f, f, &reqm_k, &imqm_k));
                    PS(m_mathOperations.add(f, f1, f));
                    mu->setZeroMatrix(v);
                    m_mathOperations.setSubColumns(0, k);
                    PS(m_mathOperations.multiply(EV, V, Q));
                    switchPointer(V, EV);
                    host::PrintMatrix("f =", f);
                    if (this->executeArnoldiFactorization(false, k - 1) == false) {
                        finish = true;
                    }
                } else {
                    finish = true;
                }
            }
            for (uintt fa = 0; fa < m_count; fa++) {
                if (m_reoutputs) {
                    m_reoutputs[fa] = wanted[fa].re;
                }
                if (m_imoutputs) {
                    m_imoutputs[fa] = wanted[fa].im;
                }
            }
        }

        void IraMethod::multiply(math::Matrix* a, math::Matrix* b,
                math::Matrix* c, bool first) {
            PS(m_mathOperations.dotProduct(a, b, c));
        }

        bool IraMethod::executeArnoldiFactorization(bool init, intt initj) {
            if (init) {
                debugFunc();
                multiply(w, A, v, true);
                debugFunc();
                mc->setVector(V, 0, v, v->rows);
                m_mathOperations.setSubRows(0, 1);
                PS(m_mathOperations.transpose(transposeV, V));
                m_mathOperations.setSubColumns(0, 1);
                PS(m_mathOperations.dotProduct(h, transposeV, w));
                PS(m_mathOperations.dotProduct(vh, V, h));
                PS(m_mathOperations.substract(f, w, vh));
                mc->setVector(H, 0, h, 1);
            }
            floatt mf = 0;
            floatt mh = 0;
            floatt B = 0;
            for (uintt j = initj; j < m_k - 1; j++) {
                PS(m_mathOperations.magnitude(&B, f));
                if (fabs(B) < MATH_VALUE_LIMIT) {
                    return false;
                }
                floatt rB = 1. / B;
                PS(m_mathOperations.multiply(v, f, &rB));
                mc->setVector(V, j + 1, v, v->rows);

                memset(&H->reValues[H->columns * (j + 1)], 0, H->columns *
                        sizeof (floatt));
                if (H->imValues) {
                    memset(&H->imValues[H->columns * (j + 1)], 0, H->columns *
                            sizeof (floatt));
                }
                H->reValues[(j) + H->columns * (j + 1)] = B;
                debugFunc();
                multiply(w, A, v, false);
                debugFunc();
                m_mathOperations.setSubRows(initj, j + 2);
                PS(m_mathOperations.transpose(transposeV, V));
                PS(m_mathOperations.dotProduct(h, transposeV, w));
                PS(m_mathOperations.dotProduct(vh, V, h));
                PS(m_mathOperations.substract(f, w, vh));
                PS(m_mathOperations.magnitude(&mf, f));
                PS(m_mathOperations.magnitude(&mh, h));
                if (mf < m_rho * mh) {
                    PS(m_mathOperations.dotProduct(s, transposeV, f));
                    m_mathOperations.setSubColumns(initj, s->rows);
                    PS(m_mathOperations.dotProduct(vs, V, s));
                    PS(m_mathOperations.substract(f, f, vs));
                    PS(m_mathOperations.add(h, h, s));
                }
                mc->setVector(H, j + 1, h, j + 2);
            }
            return true;
        }

        Status IraMethodCallback::beforeExecution() {
            for (uintt fa = 0; fa < m_threadsCount; ++fa) {
                Data* thread = new Data();
                m_threads.push_back(thread);
            }
            return IraMethod::beforeExecution();
        }

        Status IraMethodCallback::afterExecution() {
            for (uintt fa = 0; fa < m_threadsCount; ++fa) {
                delete m_threads[fa];
            }
            return IraMethod::afterExecution();
        }

        floatt IraMethod::getLargestDiagonal(math::Matrix* H) const {
            floatt output = H->reValues[0];
            for (uintt fa = 1; fa < H->columns; ++fa) {
                floatt v = H->reValues[fa * H->columns + fa];
                if (v > output) {
                    output = v;
                }
            }
            return output;
        }

        floatt IraMethod::getSmallestDiagonal(math::Matrix* H) const {
            floatt output = H->reValues[0];
            for (uintt fa = 1; fa < H->columns; ++fa) {
                floatt v = H->reValues[fa * H->columns + fa];
                if (v < output) {
                    output = v;
                }
            }
            return output;
        }

        bool wayToSort(const Complex& i, const Complex& j) {
            floatt m1 = i.re * i.re + i.im * i.im;
            floatt m2 = j.re * j.re + j.im * j.im;
            return i.re < j.re;
        }

        void IraMethod::calculateH(int unwantedCount) {
            std::vector<Complex> values;
            host::CopyMatrix(H1, H);
            m_matrixModule->getMatrixUtils()->setIdentityMatrix(Q);
            m_matrixModule->getMatrixUtils()->setIdentityMatrix(QJ);
            m_matrixModule->getMatrixUtils()->setIdentityMatrix(I);
            host::PrintMatrix("H1 =", H1);
            for (uintt fa = 0; IsTriangular(H1, H1->columns - 1) == false; ++fa) {
                floatt red = 0;
                if (H1->reValues) {
                    red = H1->reValues[H1->columns * H1->rows - 1];
                }
                floatt imd = 0;
                if (H1->imValues) {
                    imd = H1->imValues[H1->columns * H1->rows - 1];
                }
                m_matrixModule->getMatrixUtils()->setDiagonalMatrix(I, red, imd);
                m_mathOperations.substract(H1, H1, I);
                //host::PrintMatrix("H1 =", H1);
                //host::PrintMatrix("I =", I);
                m_mathOperations.qrDecomposition(Q1, R1, H1);
                m_mathOperations.multiply(H1, R1, Q1);
                m_mathOperations.add(H1, H1, I);
                m_mathOperations.multiply(Q, QJ, Q1);
                switchPointer(Q, QJ);
            }
            host::PrintMatrix("H1= ", H1);
            host::PrintMatrix("Q= ", Q);
            int index = 0;
            math::Matrix* q = host::NewMatrix(1, Q->rows, 0);
            math::Matrix* q1 = host::NewMatrix(1, Q->rows, 0);
            math::Matrix* q2 = host::NewMatrix(1, Q->rows, 0);
            mc->getVector(q, q->rows, Q, index);
            host::PrintMatrix("q =", q);
            m_mathOperations.multiply(q1, H, q);
            if (H1->imValues) {
                m_mathOperations.multiply(q2, q, &H1->reValues[index * H1->columns + index],
                        &H1->imValues[index * H1->columns + index]);
            } else {
                m_mathOperations.multiply(q2, q, &H1->reValues[index * H1->columns + index]);
            }
            host::PrintMatrix("q1= ", q1);
            host::PrintMatrix("q2= ", q2);
            switchPointer(Q, QJ);
            notSorted.clear();
            for (uintt fa = 0; fa < H1->columns; ++fa) {
                floatt rev = getReDiagonal(H1, fa);
                floatt imv = getImDiagonal(H1, fa);
                Complex c;
                c.re = rev;
                c.im = imv;
                values.push_back(c);
                notSorted.push_back(c);
            }
            std::sort(values.begin(), values.end(), wayToSort);
            for (uintt fa = 0; fa < values.size(); ++fa) {
                Complex value = values[fa];
                if (fa < unwantedCount) {
                    unwanted.push_back(value);
                } else {
                    wanted.push_back(value);
                    for (uintt fb = 0; fb < notSorted.size(); ++fb) {
                        if (notSorted[fb].im == value.im &&
                                notSorted[fb].re == value.re) {
                            wantedIndecies.push_back(fb);
                        }
                    }
                }
            }
        }

        IraMethodCallback::IraMethodCallback(MathOperations* mathOperations,
                uintt realCount) :
        IraMethod(mathOperations) {
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

        IraMethodCallback::~IraMethodCallback() {
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

        void IraMethodCallback::preMultiply(math::Matrix* a, math::Matrix* b,
                math::Matrix* c) {
            fprintf(stderr, "%s %s %d \n", __FUNCTION__, __FILE__, __LINE__);
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
                            fprintf(stderr, "%s %s %d \n", __FUNCTION__, __FILE__, __LINE__);
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

            fprintf(stderr, "%s %s %d \n", __FUNCTION__, __FILE__, __LINE__);
            isFinish = true;
        }

        void IraMethodCallback::Data::calculate(uintt index, uintt count,
                uintt begin, uintt size, uint count2) {
            uintt diff = size / count;
            uintt dindex = diff*index;
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

        void IraMethodCallback::ThreadFunction(void* ptr) {
            Data* data = (Data*) ptr;
            uintt m_index = data->m_beginIndex;
            for (uintt fa1 = data->m_brow; fa1 < data->m_erow; ++fa1) {
                fprintf(stderr, "index =  %u %u %u %u \n", m_index, data->m_count2, data->m_brow, data->m_erow);
                floatt rev = 0;
                floatt imv = 0;
                for (uintt fa = 0; fa < data->m_count2; ++fa) {
                    rev += data->m_reoutputs[m_index] * data->m_reoutputs1[m_index] -
                            data->m_imoutputs[m_index] * data->m_imoutputs1[m_index];
                    imv += data->m_reoutputs[m_index] * data->m_imoutputs1[m_index] +
                            data->m_imoutputs[m_index] * data->m_reoutputs1[m_index];
                    m_index++;
                }
                if (data->w->reValues) {
                    data->w->reValues[fa1] = rev;
                    fprintf(stderr, "%f %u \n", rev, fa1);
                    fprintf(stderr, "index1 =  %u \n", m_index);
                }
                if (data->w->imValues) {
                    data->w->imValues[fa1] = imv;
                }
            }
            return;
        }

        void IraMethodCallback::postMultiply(math::Matrix* a, math::Matrix* b,
                math::Matrix* c) {
            math::Matrix* w = a;
            for (uintt fa = 0; fa < m_threads.size(); ++fa) {
                m_threads[fa]->calculate(fa, m_threadsCount,
                        m_rows, m_rows1 - m_rows, m_count2);
                m_threads[fa]->m_reoutputs = m_reoutputs;
                m_threads[fa]->m_reoutputs1 = m_reoutputs1;
                m_threads[fa]->m_imoutputs = m_imoutputs;
                m_threads[fa]->m_imoutputs1 = m_imoutputs1;
                m_threads[fa]->w = w;
                m_threads[fa]->thread.setFunction(ThreadFunction,
                        m_threads[fa]);
                m_threads[fa]->thread.run(fa == 1);
            }
            for (uintt fa = 0; fa < m_threads.size(); ++fa) {
                m_threads[fa]->thread.yield();
            }
            fprintf(stderr, "row = %u %u \n", m_rows, m_rows1);
            m_rows = m_rows1;
        }

        void IraMethodCallback::multiply(math::Matrix* a, math::Matrix* b,
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
                    invokeCallbacks(EVENT_MATRIX_MULTIPLICATION, m_event);
                    postMultiply(a, b, c);
                }
            } else {
                host::CopyMatrix(a, b);
            }
        }

        const int IraMethodCallback::EVENT_MATRIX_MULTIPLICATION = 0;

        IraMethodCallback::Event::Event(floatt* reoutputs,
                floatt* imoutputs,
                uintt* matrixEntries, uintt count) {
            m_reoutputs = reoutputs;
            m_imoutputs = imoutputs;
            m_matrixEntries = matrixEntries;
            m_count = count;
        }

        IraMethodCallback::Event::Event() {
            m_reoutputs = NULL;
            m_imoutputs = NULL;
            m_matrixEntries = NULL;
            m_count = 0;
        }

        IraMethodCallback::Event::~Event() {

        }

        void IraMethodCallback::Event::setPointers(floatt* reoutputs,
                floatt* imoutputs,
                uintt* matrixEntries) {
            m_reoutputs = reoutputs;
            m_imoutputs = imoutputs;
            m_matrixEntries = matrixEntries;
        }

        void IraMethodCallback::Event::setCount(uintt count) {
            m_count = count;
            memset(m_reoutputs, 0, count * sizeof (floatt));
            memset(m_imoutputs, 0, count * sizeof (floatt));
        }

        uintt IraMethodCallback::Event::getCount() const {
            return m_count;
        }

        floatt* IraMethodCallback::Event::getReOutputs() const {
            return m_reoutputs;
        }

        floatt* IraMethodCallback::Event::getImOutputs() const {
            return m_imoutputs;
        }

        uintt* IraMethodCallback::Event::getMatrixEntries() const {
            return m_matrixEntries;
        }
    }
}