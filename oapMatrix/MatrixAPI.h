#ifndef OAP_MATRIXAPI_H
#define OAP_MATRIXAPI_H

#include "Matrix.h"
#include "MatrixTestAPI.h"
#include "CuCore.h"
#include "Dim3.h"
#include "assert.h"

#ifdef CUDA

__hostdeviceinline__ void SetRe(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->reValues[c + r * m->columns] = v;
}

__hostdeviceinline__ floatt GetRe(const math::Matrix* m, uintt c, uintt r) {
  return m->reValues[c + r * m->columns];
}

__hostdeviceinline__ floatt GetReIndex(const math::Matrix* m, uintt index) {
  return m->reValues[index];
}

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->imValues[c + r * m->columns] = v;
}

__hostdeviceinline__ floatt GetIm(const math::Matrix* m, uintt c, uintt r) {
  return m->imValues[c + r * m->columns];
}

__hostdeviceinline__ floatt GetImIndex(const math::Matrix* m, uintt index) {
  return m->imValues[index];
}

__hostdeviceinline__ void Reset(math::Matrix* m) {}

__hostdeviceinline__ void Push(math::Matrix* m) {}

__hostdeviceinline__ void Pop(math::Matrix* m) {}

#else

#include "MatrixTestAPI.h"

__hostdeviceinline__ void SetRe(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->reValues[c + r * m->columns] = v;
  test::setRe(m, c, r, v);
  assert(c + r * m->columns < m->realRows * m->realColumns);
}

__hostdeviceinline__ floatt GetRe(const math::Matrix* m, uintt c, uintt r) {
  test::getRe(m, c, r, m->reValues[c + r * m->columns]);
  assert(c + r * m->columns < m->realRows * m->realColumns);
  return m->reValues[c + r * m->columns];
}

__hostdeviceinline__ floatt GetReIndex(const math::Matrix* m, uintt index) {
  test::getRe(m, index, m->reValues[index]);
  assert(index < m->realRows * m->realColumns);
  return m->reValues[index];
}

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->imValues[c + r * m->columns] = v;
  test::setIm(m, c, r, v);
  assert(c + r * m->columns < m->realRows * m->realColumns);
}

__hostdeviceinline__ floatt GetIm(const math::Matrix* m, uintt c, uintt r) {
  test::getIm(m, c, r, m->imValues[c + r * m->columns]);
  assert(c + r * m->columns < m->realRows * m->realColumns);
  return m->imValues[c + r * m->columns];
}

__hostdeviceinline__ floatt GetImIndex(const math::Matrix* m, uintt index) {
  test::getIm(m, index, m->imValues[index]);
  assert(index < m->realRows * m->realColumns);
  return m->imValues[index];
}

__hostdeviceinline__ void Reset(math::Matrix* m) {
  HOST_INIT();
  //threads_sync();
  test::reset(m);
  //threads_sync();
}

__hostdeviceinline__ void Push(math::Matrix* m) {
  HOST_INIT();
  //threads_sync();
  test::push(m);
  //threads_sync();
}

__hostdeviceinline__ void Pop(math::Matrix* m) {
  HOST_INIT();
  //threads_sync();
  test::pop(m);
  //threads_sync();
}

#endif

#endif  // MATRIXAPI_H
