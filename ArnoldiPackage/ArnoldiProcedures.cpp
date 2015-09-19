#include <math.h>
#include <algorithm>
#include "ArnoldiProcedures.h"
#include "DeviceMatrixModules.h"
#include "Callbacks.h"

const char* kernelsFiles[] = {"liboglaMatrixCuda.cubin", NULL};
namespace ArnUtils {

bool SortLargestValues(const Complex& i, const Complex& j) {
  floatt m1 = i.re * i.re + i.im * i.im;
  floatt m2 = j.re * j.re + j.im * j.im;
  return m1 < m2;
}

bool SortLargestReValues(const Complex& i, const Complex& j) {
  return i.re < j.re;
}

bool SortLargestImValues(const Complex& i, const Complex& j) {
  return i.im < j.im;
}

bool SortSmallestValues(const Complex& i, const Complex& j) {
  floatt m1 = i.re * i.re + i.im * i.im;
  floatt m2 = j.re * j.re + j.im * j.im;
  return m1 > m2;
}

bool SortSmallestReValues(const Complex& i, const Complex& j) {
  return i.re > j.re;
}

bool SortSmallestImValues(const Complex& i, const Complex& j) {
  return i.im > j.im;
}

MatrixInfo::MatrixInfo() : isRe(false), isIm(false) {
  m_matrixDim.columns = 0;
  m_matrixDim.rows = 0;
}

MatrixInfo::MatrixInfo(bool _isRe, bool _isIm, uintt _columns, uintt _rows)
    : isRe(_isRe), isIm(_isIm) {
  m_matrixDim.columns = _columns;
  m_matrixDim.rows = _rows;
}
}

CuHArnoldi::CuHArnoldi()
    : m_wasAllocated(false),
      m_k(0),
      m_sortType(NULL),
      w(NULL),
      f(NULL),
      f1(NULL),
      vh(NULL),
      h(NULL),
      s(NULL),
      vs(NULL),
      V(NULL),
      transposeV(NULL),
      V1(NULL),
      V2(NULL),
      H(NULL),
      HC(NULL),
      H1(NULL),
      H2(NULL),
      A1(NULL),
      A2(NULL),
      I(NULL),
      v(NULL),
      QT(NULL),
      Q1(NULL),
      Q2(NULL),
      R1(NULL),
      R2(NULL),
      HO(NULL),
      HO1(NULL),
      Q(NULL),
      QJ(NULL),
      q(NULL),
      q1(NULL),
      q2(NULL),
      GT(NULL),
      G(NULL),
      EV(NULL),
      EV1(NULL),
      EQ1(NULL),
      EQ2(NULL),
      EQ3(NULL) {
  m_kernel.load(kernelsFiles);
  m_calculateTriangularHPtr = NULL;
}

CuHArnoldi::~CuHArnoldi() {
  dealloc1();
  dealloc2();
  dealloc3();
  m_kernel.unload();
}

void CuHArnoldi::calculateTriangularH() {
  bool status = false;
  m_cuMatrix.setIdentity(Q1);
  status = m_cuMatrix.isUpperTriangular(H1);
  uintt fb = 0;
  CudaUtils::PrintMatrix("H1=", H1, false, false);
  for (; fb < 20000; ++fb) {
    m_cuMatrix.QR(Q, R1, H1, Q2, R2, G, GT);
    CudaUtils::PrintMatrix("H=", H1, false, false);
    CudaUtils::PrintMatrix("Q=", Q);
    CudaUtils::PrintMatrix("R=", R1);
    m_cuMatrix.dotProduct(H1, R1, Q);
    m_cuMatrix.dotProduct(QJ, Q, Q1);
    switchPointer(&QJ, &Q1);
    status = m_cuMatrix.isUpperTriangular(H1);
    //if (fb == 200) {abort();}
  }
  if (fb & 1 == 0) {
    device::CopyDeviceMatrixToDeviceMatrix(Q, Q1);
  } else {
    device::CopyDeviceMatrixToDeviceMatrix(Q, QJ);
  }
}

void CuHArnoldi::calculateTriangularHInDevice() {
  CudaUtils::PrintMatrix("BH1", H1, false, false);
  void* params[] = {&H1, &Q, &R1, &Q1, &QJ, &Q2, &R2, &G, &GT};
  m_kernel.setDimensions(m_Hcolumns, m_Hrows);
  device::Kernel::Execute("CUDAKernel_CalculateTriangularH", params, m_kernel);
  CudaUtils::PrintMatrix("AH1", H1);
}

void CuHArnoldi::setSortType(ArnUtils::SortType sortType) {
  m_sortType = sortType;
}

void aux_switchPointer(math::Matrix** a, math::Matrix** b) {
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}

void CuHArnoldi::calculateTriangularHEigens(uintt unwantedCount) {
  std::vector<Complex> values;
  device::CopyDeviceMatrixToDeviceMatrix(H1, H);
  m_cuMatrix.setIdentity(Q);
  m_cuMatrix.setIdentity(QJ);
  m_cuMatrix.setIdentity(I);
  (this->*m_calculateTriangularHPtr)();
  int index = 0;
  m_cuMatrix.getVector(q, m_qrows, Q, index);
  m_cuMatrix.dotProduct(q1, H, q);
  if (m_matrixInfo.isIm) {
    uintt index1 = index * m_H1columns + index;
    floatt re = CudaUtils::GetReValue(H1, index1);
    floatt im = CudaUtils::GetImValue(H1, index1);
    m_cuMatrix.multiplyConstantMatrix(q2, q, re, im);
  } else {
    uintt index1 = index * m_H1columns + index;
    floatt re = CudaUtils::GetReValue(H1, index1);
    m_cuMatrix.multiplyConstantMatrix(q2, q, re);
  }
  aux_switchPointer(&Q, &QJ);
  notSorted.clear();
  for (uintt fa = 0; fa < m_H1columns; ++fa) {
    floatt rev = CudaUtils::GetReDiagonal(H1, fa);
    floatt imv = CudaUtils::GetImDiagonal(H1, fa);
    Complex c(rev, imv);
    values.push_back(c);
    notSorted.push_back(c);
  }
  std::sort(values.begin(), values.end(), m_sortType);
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
}

bool CuHArnoldi::executeArnoldiFactorization(bool init, intt initj,
                                             MatrixEx** dMatrixEx,
                                             floatt m_rho) {
  if (init) {
    multiply(w, v);
    m_cuMatrix.setVector(V, 0, v, m_vrows);
    m_cuMatrix.transposeMatrixEx(transposeV, V, dMatrixEx[0]);
    m_cuMatrix.dotProductEx(h, transposeV, w, dMatrixEx[1]);
    m_cuMatrix.dotProduct(vh, V, h);
    m_cuMatrix.substract(f, w, vh);
    m_cuMatrix.setVector(H, 0, h, 1);
  }
  floatt mf = 0;
  floatt mh = 0;
  floatt B = 0;
  for (uintt fa = initj; fa < m_k - 1; ++fa) {
    m_cuMatrix.magnitude(B, f);
    if (fabs(B) < MATH_VALUE_LIMIT) {
      return false;
    }
    floatt rB = 1. / B;
    m_cuMatrix.multiplyConstantMatrix(v, f, rB);
    fprintf(stderr, "rB = %f \n", rB);
    fprintf(stderr, "B = %f \n", B);
    m_cuMatrix.setVector(V, fa + 1, v, m_vrows);
    CudaUtils::SetZeroRow(H, fa + 1, true, true);
    CudaUtils::SetReValue(H, (fa) + m_Hcolumns * (fa + 1), B);
    multiply(w, v);
    MatrixEx matrixEx = {0, m_transposeVcolumns, initj, fa + 2, 0, 0};
    device::SetMatrixEx(dMatrixEx[2], &matrixEx);
    m_cuMatrix.transposeMatrixEx(transposeV, V, dMatrixEx[2]);
    m_cuMatrix.dotProduct(h, transposeV, w);
    m_cuMatrix.dotProduct(vh, V, h);
    m_cuMatrix.substract(f, w, vh);
    m_cuMatrix.magnitude(mf, f);
    m_cuMatrix.magnitude(mh, h);
    if (mf < m_rho * mh) {
      m_cuMatrix.dotProductEx(s, transposeV, f, dMatrixEx[3]);
      m_cuMatrix.dotProductEx(vs, V, s, dMatrixEx[4]);
      m_cuMatrix.substract(f, f, vs);
      m_cuMatrix.add(h, h, s);
    }
    m_cuMatrix.setVector(H, fa + 1, h, fa + 2);
  }
  return true;
}

void CuHArnoldi::initVvector() {
  CudaUtils::SetReValue(V, 0, 1.f);
  CudaUtils::SetReValue(v, 0, 1.f);
}

bool CuHArnoldi::continueProcedure() { return true; }

void CuHArnoldi::setRho(floatt rho) { m_rho = rho; }

void CuHArnoldi::setOutputs(math::Matrix* outputs) {
  m_outputs = outputs;
  m_outputsType = ArnUtils::HOST;
}

void CuHArnoldi::execute(uintt k, uintt wantedCount,
                         const ArnUtils::MatrixInfo& matrixInfo,
                         ArnUtils::Type matrixType) {
  debugAssert(wantedCount != 0);
  if (m_sortType == NULL) {
    m_sortType = ArnUtils::SortSmallestValues;
    debug("Warning! Sort type is null. Set as smallest value.");
  }

  setCalculateTriangularHPtr(k);

  const uintt dMatrixExCount = 5;
  MatrixEx** dMatrixExs = device::NewDeviceMatrixEx(dMatrixExCount);
  alloc(matrixInfo, k);

  m_matrixInfo = matrixInfo;
  m_matrixType = matrixType;

  initVvector();
  bool status = false;
  {
    const uintt initj = 0;
    uintt buffer[] = {0, m_transposeVcolumns, 0, 1, 0, 0, 0, 1, 0, m_hrows, 0,
                      m_transposeVcolumns, 0, 0, 0, 0, 0, 0, 0, m_scolumns,
                      initj, initj + 2, 0, m_transposeVcolumns, 0, m_vscolumns,
                      0, m_vsrows, initj, initj + 2};
    device::SetMatrixEx(dMatrixExs, buffer, dMatrixExCount);
    status = executeArnoldiFactorization(true, 0, dMatrixExs, m_rho);
  }
  for (intt fax = 0; fax == 0 || status == true; ++fax) {
    unwanted.clear();
    wanted.clear();
    wantedIndecies.clear();
    calculateTriangularHEigens(k - wantedCount);
    if (status == true) {
      m_cuMatrix.setIdentity(Q);
      m_cuMatrix.setIdentity(QJ);
      uintt p = m_outputs->columns - wantedCount;
      uintt k = wantedCount;
      for (intt fa = 0; fa < p; ++fa) {
        m_cuMatrix.setDiagonal(I, unwanted[fa].re, unwanted[fa].im);
        m_cuMatrix.substract(I, H, I);
        m_cuMatrix.QR(Q1, R1, I, Q, R2, G, GT);
        m_cuMatrix.transposeMatrix(QT, Q1);
        m_cuMatrix.dotProduct(HO, H, Q1);
        m_cuMatrix.dotProduct(H, QT, HO);
        m_cuMatrix.dotProduct(Q, QJ, Q1);
        aux_switchPointer(&Q, &QJ);
      }
      aux_switchPointer(&Q, &QJ);
      m_cuMatrix.dotProduct(EV, V, Q);
      aux_switchPointer(&V, &EV);
      floatt reqm_k = CudaUtils::GetReValue(Q, m_Qcolumns * (m_Qrows - 1) + k);
      floatt imqm_k = 0;
      if (m_matrixInfo.isIm) {
        imqm_k = CudaUtils::GetImValue(Q, m_Qcolumns * (m_Qrows - 1) + k);
      }
      floatt reBm_k = CudaUtils::GetReValue(H, m_Hcolumns * (k + 1) + k);
      floatt imBm_k = 0;
      if (m_matrixInfo.isIm) {
        imBm_k = CudaUtils::GetImValue(H, m_Hcolumns * (k + 1) + k);
      }
      m_cuMatrix.getVector(v, m_vrows, V, k);
      m_cuMatrix.multiplyConstantMatrix(f1, v, reBm_k, imBm_k);
      m_cuMatrix.multiplyConstantMatrix(f, f, reqm_k, imqm_k);
      m_cuMatrix.add(f, f1, f);
      m_cuMatrix.setZeroMatrix(v);

      {
        const uintt initj = k - 1;
        uintt buffer[] = {0, m_transposeVcolumns, 0, 1, 0, 0, 0, 1, 0, m_hrows,
                          0, m_transposeVcolumns, 0, 0, 0, 0, 0, 0, 0,
                          m_scolumns, initj, initj + 2, 0, m_transposeVcolumns,
                          0, m_vscolumns, 0, m_vsrows, initj, initj + 2};
        device::SetMatrixEx(dMatrixExs, buffer, dMatrixExCount);
        status = executeArnoldiFactorization(false, k - 1, dMatrixExs, m_rho);
      }
    }
  }
  for (uintt fa = 0; fa < m_outputs->columns; fa++) {
    if (NULL != m_outputs->reValues) {
      m_outputs->reValues[fa] = wanted[fa].re;
    }
    if (NULL != m_outputs->imValues) {
      m_outputs->imValues[fa] = wanted[fa].im;
    }
  }
  device::DeleteDeviceMatrixEx(dMatrixExs);
}

bool CuHArnoldi::shouldBeReallocated(const ArnUtils::MatrixInfo& m1,
                                     const ArnUtils::MatrixInfo& m2) const {
  return m1.isIm != m2.isIm || m1.isRe != m2.isRe;
}

void CuHArnoldi::alloc(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  if (!m_wasAllocated || shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      matrixInfo.m_matrixDim.rows != m_matrixInfo.m_matrixDim.rows) {
    dealloc1();
    alloc1(matrixInfo, k);
    m_vsrows = matrixInfo.m_matrixDim.rows;
    m_vscolumns = 1;
  }
  if (!m_wasAllocated || shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      matrixInfo.m_matrixDim.rows != m_matrixInfo.m_matrixDim.rows ||
      m_k != k) {
    dealloc2();
    alloc2(matrixInfo, k);
    m_transposeVcolumns = matrixInfo.m_matrixDim.rows;
  }
  if (!m_wasAllocated || shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      m_k != k) {
    dealloc3();
    alloc3(matrixInfo, k);
    m_hrows = k;
    m_scolumns = 1;
    m_Hcolumns = k;
    m_Hrows = k;
    m_H1columns = k;
    m_Qcolumns = k;
    m_Qrows = k;
    m_qrows = k;
    m_wasAllocated = true;
    m_k = k;
  }
  CudaUtils::SetZeroMatrix(v);
  CudaUtils::SetZeroMatrix(V);
}

void CuHArnoldi::alloc1(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  w = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                            matrixInfo.m_matrixDim.rows);
  v = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                            matrixInfo.m_matrixDim.rows);
  m_vrows = matrixInfo.m_matrixDim.rows;
  f = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                            matrixInfo.m_matrixDim.rows);
  f1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                             matrixInfo.m_matrixDim.rows);
  vh = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                             matrixInfo.m_matrixDim.rows);
  vs = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                             matrixInfo.m_matrixDim.rows);
  EQ1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                              matrixInfo.m_matrixDim.rows);
  EQ2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                              matrixInfo.m_matrixDim.rows);
  EQ3 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                              matrixInfo.m_matrixDim.rows);
}

void CuHArnoldi::alloc2(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  V = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                            matrixInfo.m_matrixDim.rows);
  V1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                             matrixInfo.m_matrixDim.rows);
  V2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                             matrixInfo.m_matrixDim.rows);
  EV = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                             matrixInfo.m_matrixDim.rows);
  EV1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                              matrixInfo.m_matrixDim.rows);
  transposeV = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm,
                                     matrixInfo.m_matrixDim.rows, k);
}

void CuHArnoldi::alloc3(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  h = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  s = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  H = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  G = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  GT = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  HO = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  H1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  Q1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  Q2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  QT = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  R1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  R2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  QJ = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  I = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  Q = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  q = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  q1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  q2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
}

void CuHArnoldi::dealloc1() {
  device::DeleteDeviceMatrix(w);
  device::DeleteDeviceMatrix(v);
  device::DeleteDeviceMatrix(f);
  device::DeleteDeviceMatrix(f1);
  device::DeleteDeviceMatrix(vh);
  device::DeleteDeviceMatrix(vs);
  device::DeleteDeviceMatrix(EQ1);
  device::DeleteDeviceMatrix(EQ2);
  device::DeleteDeviceMatrix(EQ3);
}

void CuHArnoldi::dealloc2() {
  device::DeleteDeviceMatrix(V);
  device::DeleteDeviceMatrix(V1);
  device::DeleteDeviceMatrix(V2);
  device::DeleteDeviceMatrix(EV);
  device::DeleteDeviceMatrix(EV1);
  device::DeleteDeviceMatrix(transposeV);
}

void CuHArnoldi::dealloc3() {
  device::DeleteDeviceMatrix(h);
  device::DeleteDeviceMatrix(s);
  device::DeleteDeviceMatrix(H);
  device::DeleteDeviceMatrix(G);
  device::DeleteDeviceMatrix(GT);
  device::DeleteDeviceMatrix(HO);
  device::DeleteDeviceMatrix(H1);
  device::DeleteDeviceMatrix(Q1);
  device::DeleteDeviceMatrix(Q2);
  device::DeleteDeviceMatrix(QT);
  device::DeleteDeviceMatrix(R1);
  device::DeleteDeviceMatrix(R2);
  device::DeleteDeviceMatrix(QJ);
  device::DeleteDeviceMatrix(I);
  device::DeleteDeviceMatrix(Q);
  device::DeleteDeviceMatrix(q1);
  device::DeleteDeviceMatrix(q2);
}

void CuHArnoldiDefault::multiply(math::Matrix* w, math::Matrix* v) {
  m_cuMatrix.dotProduct(w, m_A, v);
}

void CuHArnoldiCallback::multiply(math::Matrix* w, math::Matrix* v) {
  m_multiplyFunc(w, v, m_userData);
}

void CuHArnoldiCallback::setCallback(
    CuHArnoldiCallback::MultiplyFunc multiplyFunc, void* userData) {
  m_multiplyFunc = multiplyFunc;
  m_userData = userData;
}
