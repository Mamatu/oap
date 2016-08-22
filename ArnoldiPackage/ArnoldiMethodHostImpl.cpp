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



#include <math.h>
#include <stdio.h>
#include "MathOperationsCpu.h"
#include "ArnoldiMethodHostImpl.h"
#include "MathOperationsCpu.h"

#ifdef DEBUG
#define PRINT_STATUS(d)                    \
  if (d != 0) {                            \
    fprintf(stderr, "Status == %d \n", d); \
    abort();                               \
  }
#else
#define PRINT_STATUS(d) d
#endif

#define MIN_VALUE 0.001

namespace math {

inline void switchPointer(math::Matrix*& a, math::Matrix*& b) {
  math::Matrix* temp = b;
  b = a;
  a = temp;
}

bool IsTriangular(math::Matrix* matrix, uintt count) {
  uintt index = 0;
  for (uintt fa = 0; fa < matrix->columns - 1; ++fa) {
    floatt revalue = matrix->reValues[fa + matrix->columns * (fa + 1)];
    floatt imvalue = 0;
    if (NULL != matrix->imValues) {
      imvalue = matrix->imValues[fa + matrix->columns * (fa + 1)];
    }
    if ((-MIN_VALUE < revalue && revalue < MIN_VALUE) &&
        (-MIN_VALUE < imvalue && imvalue < MIN_VALUE)) {
      index++;
    }
  }
  return index >= count;
}

void CalculateTriangular(MathOperationsCpu* m_operations,
                         MatrixModule* m_module, math::Matrix* H,
                         math::Matrix* H1, math::Matrix* Q, math::Matrix* QJ,
                         math::Matrix* Q1, math::Matrix* R1, math::Matrix* I) {
  host::CopyMatrix(H1, H);
  m_module->getMatrixUtils()->setIdentityMatrix(Q);
  m_module->getMatrixUtils()->setIdentityMatrix(QJ);
  m_module->getMatrixUtils()->setIdentityMatrix(I);
  for (uintt fa = 0; IsTriangular(H1, H1->columns - 1) == false && fa < 10000;
       ++fa) {
#if 0
        floatt red = 0;
        if (H1->reValues) {
            red = H1->reValues[H1->columns * H1->rows - 1];
        }
        floatt imd = 0;
        if (H1->imValues) {
            imd = H1->imValues[H1->columns * H1->rows - 1];
        }
        m_module->getMatrixUtils()->setDiagonalMatrix(I, red, imd);
        m_operations->substract(H1, H1, I);
#endif
    m_operations->qrDecomposition(Q1, R1, H1);
    m_operations->multiply(H1, R1, Q1);
#if 0
        m_operations->add(H1, H1, I);
#endif
    m_operations->multiply(Q, QJ, Q1);
    switchPointer(Q, QJ);
  }
}

ArnoldiMethodCpu::ArnoldiMethodCpu(MatrixModule* matrixModule,
                                   MathOperationsCpu* mathOperations)
    : IArnoldiMethod(matrixModule), m_operations(mathOperations) {
  this->m_rho = 1. / 3.14;
  this->m_k = 0;
  this->m_wantedCount = 0;
  diff = -10.552;
}

math::Matrix* ArnoldiMethodCpu::getV() const { return this->v; }

math::Matrix* ArnoldiMethodCpu::getW() const { return this->w; }

math::Matrix* ArnoldiMethodCpu::getA() const { return this->A; }

ArnoldiMethodCpu::ArnoldiMethodCpu(MathOperationsCpu* mathOperations)
    : IArnoldiMethod(HostMatrixModules::GetInstance()),
      m_operations(mathOperations) {
  this->m_rho = 1. / 3.14;
  this->m_k = 0;
  this->m_wantedCount = 0;
  m_oldA = NULL;
}

void ArnoldiMethodCpu::setHSize(uintt k) { this->m_k = k; }

void ArnoldiMethodCpu::setRho(floatt rho) { this->m_rho = rho; }

ArnoldiMethodCpu::~ArnoldiMethodCpu() { dealloc(); }

bool ArnoldiMethodCpu::testProcedure(uintt fa) {
  uintt index = wantedIndecies[fa];
  Complex b = notSorted[index];
  m_copier->getVector(v, v->rows, EV, index);
  multiply(EQ1, A, v, false);
  m_operations->multiply(EQ2, v, &(b.re), &(b.im));
  floatt d = 0;
  m_operations->substract(EQ3, EQ1, EQ2);
  // host::PrintReMatrix("EQ1 =", EQ1);
  // host::PrintReMatrix("EQ2 =", EQ2);
  m_operations->magnitude(&d, EQ3);
  if (d < diff || diff < 0) {
    diff = d;
    //    fprintf(stderr, "diff = %f \n", diff);
    //    fprintf(stderr, "ev = %f %f \n1", b.re, b.im);
  }
  // fprintf(stderr, "diA = %f \n", d);
  // fprintf(stderr, "evA = %f %f \n", b.re, b.im);
}

bool ArnoldiMethodCpu::continueProcedure() {
  m_operations->multiply(EV, V, Q);
  // host::PrintReMatrix("V =", V);
  // host::PrintReMatrix("Q =", Q);
  for (uintt fa = 0; fa < wanted.size(); ++fa) {
    testProcedure(fa);
  }
  return true;
}

floatt ArnoldiMethodCpu::getReDiagonal(math::Matrix* matrix, intt index) {
  if (matrix->reValues == NULL) {
    return 0;
  }
  return matrix->reValues[index + matrix->columns * index];
}

floatt ArnoldiMethodCpu::getImDiagonal(math::Matrix* matrix, intt index) {
  if (matrix->imValues == NULL) {
    return 0;
  }
  return matrix->imValues[index + matrix->columns * index];
}

bool ArnoldiMethodCpu::isEigenValue(math::Matrix* matrix, intt index) {
  floatt v = matrix->reValues[(index - 1) + matrix->columns * index];
  if (fabs(v) < MATH_VALUE_LIMIT) {
    return true;
  }
  return false;
}

void ArnoldiMethodCpu::alloc(math::Matrix* A) {
  if (m_oldA == NULL ||
      (A->rows != m_oldA->rows || A->columns != m_oldA->columns)) {
    dealloc();
    debugAssert(m_k != 0);
    w = m_module->newMatrix(A, 1, A->rows);
    v = m_module->newMatrix(A, 1, A->rows);
    f = m_module->newMatrix(A, 1, A->rows);
    f1 = m_module->newMatrix(A, 1, A->rows);
    vh = m_module->newMatrix(A, 1, A->rows);
    h = m_module->newMatrix(A, 1, m_k);
    s = m_module->newMatrix(A, 1, m_k);
    vs = m_module->newMatrix(A, 1, A->rows);
    V = m_module->newMatrix(A, m_k, A->rows);
    V1 = m_module->newMatrix(A, m_k, A->rows);
    V2 = m_module->newMatrix(A, m_k, A->rows);
    EQ1 = m_module->newMatrix(A, 1, A->rows);
    EQ2 = m_module->newMatrix(A, 1, A->rows);
    EQ3 = m_module->newMatrix(A, 1, A->rows);
    EV = m_module->newMatrix(A, m_k, A->rows);
    EV1 = m_module->newMatrix(A, m_k, A->rows);
    H = m_module->newMatrix(A, m_k, m_k);
    HO = m_module->newMatrix(A, m_k, m_k);
    H1 = m_module->newMatrix(A, m_k, m_k);
    Q = m_module->newMatrix(A, m_k, m_k);
    Q1 = m_module->newMatrix(A, m_k, m_k);
    QT = m_module->newMatrix(A, m_k, m_k);
    R1 = m_module->newMatrix(A, m_k, m_k);
    R2 = m_module->newMatrix(A, m_k, m_k);
    QJ = m_module->newMatrix(A, m_k, m_k);
    I = m_module->newMatrix(A, m_k, m_k);
    A1 = m_module->newMatrix(A, A->columns, A->columns);
    transposeV = m_module->newMatrix(A, A->rows, m_k);
    m_oldA = A;
  }
}

void ArnoldiMethodCpu::dealloc() {
  if (m_oldA != NULL) {
    debugFunc();
    m_module->deleteMatrix(w);
    m_module->deleteMatrix(QJ);
    m_module->deleteMatrix(Q);
    m_module->deleteMatrix(f);
    m_module->deleteMatrix(f1);
    m_module->deleteMatrix(vh);
    m_module->deleteMatrix(h);
    m_module->deleteMatrix(s);
    m_module->deleteMatrix(vs);
    m_module->deleteMatrix(V);
    m_module->deleteMatrix(H);
    m_module->deleteMatrix(H1);
    m_module->deleteMatrix(I);
    m_module->deleteMatrix(v);
    m_module->deleteMatrix(transposeV);
    m_module->deleteMatrix(A1);
    m_module->deleteMatrix(V1);
    m_module->deleteMatrix(V2);
    m_module->deleteMatrix(EV);
    m_module->deleteMatrix(EV1);
    m_module->deleteMatrix(R1);
    m_module->deleteMatrix(R2);
    m_module->deleteMatrix(QT);
    m_module->deleteMatrix(Q1);
    m_module->deleteMatrix(HO);
    m_module->deleteMatrix(EQ1);
    m_module->deleteMatrix(EQ2);
    m_module->deleteMatrix(EQ3);
    debugFunc();
  }
}

void ArnoldiMethodCpu::execute() {
  m_utils = m_module->getMatrixUtils();
  m_allocator = m_module->getMatrixAllocator();
  m_copier = m_module->getMatrixCopier();

  diff = -10.552;
  m_wantedCount = m_count;
  math::Matrix* A = m_matrix;
  // m_mathOperations->setThreadsCount(m_threadsCount);
  alloc(A);
  v->reValues[0] = 1;
  floatt tempLenght = 0;
  m_operations->magnitude(&tempLenght, v);
  tempLenght = 1. / tempLenght;
  m_operations->multiply(v, v, &tempLenght);
  m_copier->setVector(V, 0, v, v->rows);
  this->A = A;
  bool finish = false;
  this->executeArnoldiFactorization();
  for (intt fax = 0; finish == false; ++fax) {
    unwanted.clear();
    wanted.clear();
    wantedIndecies.clear();
    this->calculateH(H->columns - m_wantedCount);
    if (continueProcedure() == true) {
      m_utils->setIdentityMatrix(Q);
      m_utils->setIdentityMatrix(QJ);
      int p = unwanted.size();
      int k = wanted.size();
      for (intt fa = 0; fa < p; ++fa) {
        m_utils->setDiagonalMatrix(I, unwanted[fa].re, unwanted[fa].im);
        PRINT_STATUS(m_operations->substract(I, H, I));
        PRINT_STATUS(m_operations->qrDecomposition(Q1, R1, I));
        PRINT_STATUS(m_operations->transpose(QT, Q1));
        PRINT_STATUS(m_operations->dotProduct(HO, H, Q1));
        PRINT_STATUS(m_operations->dotProduct(H, QT, HO));
        PRINT_STATUS(m_operations->dotProduct(Q, QJ, Q1));
        switchPointer(Q, QJ);
      }
      switchPointer(Q, QJ);
      PRINT_STATUS(m_operations->multiply(EV, V, Q));
      switchPointer(V, EV);
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
      m_copier->getVector(v, v->rows, V, k);
      PRINT_STATUS(m_operations->multiply(f1, v, &reBm_k, &imBm_k));
      PRINT_STATUS(m_operations->multiply(f, f, &reqm_k, &imqm_k));
      PRINT_STATUS(m_operations->add(f, f1, f));
      m_utils->setZeroMatrix(v);
      if (this->executeArnoldiFactorization(false, k - 1) == false) {
        finish = true;
      }
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

void ArnoldiMethodCpu::multiply(math::Matrix* a, math::Matrix* b,
                                math::Matrix* c, bool first) {
  PRINT_STATUS(m_operations->dotProduct(a, b, c));
}

bool ArnoldiMethodCpu::executeArnoldiFactorization(bool init, intt initj) {
  if (true == init) {
    multiply(w, A, v, true);
    m_copier->setVector(V, 0, v, v->rows);
    m_operations->setSubRows(1);
    PRINT_STATUS(m_operations->transpose(transposeV, V));
    m_operations->setSubRows(1);
    PRINT_STATUS(m_operations->dotProduct(h, transposeV, w));
    PRINT_STATUS(m_operations->dotProduct(vh, V, h));
    PRINT_STATUS(m_operations->substract(f, w, vh));
    m_copier->setVector(H, 0, h, 1);
  }
  floatt mf = 0;
  floatt mh = 0;
  floatt B = 0;
  for (uintt j = initj; j < m_k - 1; j++) {
    PRINT_STATUS(m_operations->magnitude(&B, f));
    if (fabs(B) < MATH_VALUE_LIMIT) {
      return false;
    }
    floatt rB = 1. / B;
    PRINT_STATUS(m_operations->multiply(v, f, &rB));
    m_copier->setVector(V, j + 1, v, v->rows);
    memset(&H->reValues[H->columns * (j + 1)], 0, H->columns * sizeof(floatt));
    if (H->imValues) {
      memset(&H->imValues[H->columns * (j + 1)], 0,
             H->columns * sizeof(floatt));
    }
    H->reValues[(j) + H->columns * (j + 1)] = B;
    multiply(w, A, v, false);
    m_operations->setSubRows(initj, j + 2);
    PRINT_STATUS(m_operations->transpose(transposeV, V));
    PRINT_STATUS(m_operations->dotProduct(h, transposeV, w));
    PRINT_STATUS(m_operations->dotProduct(vh, V, h));
    PRINT_STATUS(m_operations->substract(f, w, vh));
    PRINT_STATUS(m_operations->magnitude(&mf, f));
    PRINT_STATUS(m_operations->magnitude(&mh, h));
    if (mf < m_rho * mh) {
      m_operations->setSubRows(initj, j + 2);
      PRINT_STATUS(m_operations->dotProduct(s, transposeV, f));
      PRINT_STATUS(m_operations->dotProduct(vs, V, s, initj, j + 2));
      PRINT_STATUS(m_operations->substract(f, f, vs));
      PRINT_STATUS(m_operations->add(h, h, s));
    }
    m_copier->setVector(H, j + 1, h, j + 2);
  }
  return true;
}

math::Status ArnoldiMethodCallbackCpu::beforeExecution() {
  for (uintt fa = 0; fa < m_threadsCount; ++fa) {
    Data* thread = new Data();
    m_threads.push_back(thread);
  }
  return ArnoldiMethodCpu::beforeExecution();
}

math::Status ArnoldiMethodCallbackCpu::afterExecution() {
  for (uintt fa = 0; fa < m_threadsCount; ++fa) {
    delete m_threads[fa];
  }
  return ArnoldiMethodCpu::afterExecution();
}

floatt ArnoldiMethodCpu::getLargestDiagonal(math::Matrix* H) const {
  floatt output = H->reValues[0];
  for (uintt fa = 1; fa < H->columns; ++fa) {
    floatt v = H->reValues[fa * H->columns + fa];
    if (v > output) {
      output = v;
    }
  }
  return output;
}

floatt ArnoldiMethodCpu::getSmallestDiagonal(math::Matrix* H) const {
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
  return m1 < m2;
}

void ArnoldiMethodCpu::calculateH(int unwantedCount) {
  std::vector<Complex> values;
  host::CopyMatrix(H1, H);
  m_module->getMatrixUtils()->setIdentityMatrix(Q);
  m_module->getMatrixUtils()->setIdentityMatrix(QJ);
  m_module->getMatrixUtils()->setIdentityMatrix(I);
  for (uintt fa = 0; IsTriangular(H1, H1->columns - 1) == false && fa < 10000;
       ++fa) {
#if 0
        floatt red = 0;
        if (H1->reValues) {
            red = H1->reValues[H1->columns * H1->rows - 1];
        }
        floatt imd = 0;
        if (H1->imValues) {
            imd = H1->imValues[H1->columns * H1->rows - 1];
        }
        m_module->getMatrixUtils()->setDiagonalMatrix(I, red, imd);
        m_operations->substract(H1, H1, I);
#endif
    m_operations->qrDecomposition(Q1, R1, H1);
    m_operations->multiply(H1, R1, Q1);
#if 0
        m_operations->add(H1, H1, I);
#endif
    m_operations->multiply(Q, QJ, Q1);
    switchPointer(Q, QJ);
  }
  int index = 0;
  math::Matrix* q = host::NewMatrix(1, Q->rows, 0);
  math::Matrix* q1 = host::NewMatrix(1, Q->rows, 0);
  math::Matrix* q2 = host::NewMatrix(1, Q->rows, 0);
  m_copier->getVector(q, q->rows, Q, index);
  m_operations->multiply(q1, H, q);
  if (H1->imValues) {
    m_operations->multiply(q2, q, &H1->reValues[index * H1->columns + index],
                           &H1->imValues[index * H1->columns + index]);
  } else {
    m_operations->multiply(q2, q, &H1->reValues[index * H1->columns + index]);
  }
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
        if (notSorted[fb].im == value.im && notSorted[fb].re == value.re) {
          wantedIndecies.push_back(fb);
        }
      }
    }
  }
  host::DeleteMatrix(q);
  host::DeleteMatrix(q1);
  host::DeleteMatrix(q2);
}

ArnoldiMethodCallbackCpu::ArnoldiMethodCallbackCpu(
    MathOperationsCpu* mathOperations, uintt realCount)
    : ArnoldiMethodCpu(mathOperations) {
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

ArnoldiMethodCallbackCpu::~ArnoldiMethodCallbackCpu() {
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

void ArnoldiMethodCallbackCpu::preMultiply(math::Matrix* a, math::Matrix* b,
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

void ArnoldiMethodCallbackCpu::Data::calculate(uintt index, uintt count,
                                               uintt begin, uintt size,
                                               uint count2) {
  uintt diff = size / count;
  uintt dindex = diff * index;
  if (index == count - 1) {
    m_brow = begin + dindex;
    m_erow = begin + size;
  } else {
    m_brow = begin + dindex;
    m_erow = begin + diff * (index + 1);
  }
  m_beginIndex = dindex * count2;
  this->m_count2 = count2;
}

void ArnoldiMethodCallbackCpu::ThreadFunction(void* ptr) {
  Data* data = (Data*)ptr;
  uintt m_index = data->m_beginIndex;
  for (uintt fa1 = data->m_brow; fa1 < data->m_erow; ++fa1) {
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
    }
    if (data->w->imValues) {
      data->w->imValues[fa1] = imv;
    }
  }
  return;
}

void ArnoldiMethodCallbackCpu::postMultiply(math::Matrix* a, math::Matrix* b,
                                            math::Matrix* c) {
  math::Matrix* w = a;
  for (uintt fa = 0; fa < m_threads.size(); ++fa) {
    m_threads[fa]->calculate(fa, m_threadsCount, m_rows, m_rows1 - m_rows,
                             m_count2);
    m_threads[fa]->m_reoutputs = m_reoutputs;
    m_threads[fa]->m_reoutputs1 = m_reoutputs1;
    m_threads[fa]->m_imoutputs = m_imoutputs;
    m_threads[fa]->m_imoutputs1 = m_imoutputs1;
    m_threads[fa]->w = w;
    m_threads[fa]->thread.setFunction(ThreadFunction, m_threads[fa]);
    m_threads[fa]->thread.run(fa == 1);
  }
  for (uintt fa = 0; fa < m_threads.size(); ++fa) {
    m_threads[fa]->thread.yield();
  }
  m_rows = m_rows1;
}

void ArnoldiMethodCallbackCpu::multiply(math::Matrix* a, math::Matrix* b,
                                        math::Matrix* c, bool first) {
  if (first == false) {
    m_event->setPointers(m_reoutputs1, m_imoutputs1, m_matrixEntries);
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

const int ArnoldiMethodCallbackCpu::EVENT_MATRIX_MULTIPLICATION = 0;

ArnoldiMethodCallbackCpu::Event::Event(floatt* reoutputs, floatt* imoutputs,
                                       uintt* matrixEntries, uintt count) {
  m_reoutputs = reoutputs;
  m_imoutputs = imoutputs;
  m_matrixEntries = matrixEntries;
  m_count = count;
}

ArnoldiMethodCallbackCpu::Event::Event() {
  m_reoutputs = NULL;
  m_imoutputs = NULL;
  m_matrixEntries = NULL;
  m_count = 0;
}

ArnoldiMethodCallbackCpu::Event::~Event() {}

void ArnoldiMethodCallbackCpu::Event::setPointers(floatt* reoutputs,
                                                  floatt* imoutputs,
                                                  uintt* matrixEntries) {
  m_reoutputs = reoutputs;
  m_imoutputs = imoutputs;
  m_matrixEntries = matrixEntries;
}

void ArnoldiMethodCallbackCpu::Event::setCount(uintt count) {
  m_count = count;
  memset(m_reoutputs, 0, count * sizeof(floatt));
  memset(m_imoutputs, 0, count * sizeof(floatt));
}

uintt ArnoldiMethodCallbackCpu::Event::getCount() const { return m_count; }

floatt* ArnoldiMethodCallbackCpu::Event::getReOutputs() const {
  return m_reoutputs;
}

floatt* ArnoldiMethodCallbackCpu::Event::getImOutputs() const {
  return m_imoutputs;
}

uintt* ArnoldiMethodCallbackCpu::Event::getMatrixEntries() const {
  return m_matrixEntries;
}
}
