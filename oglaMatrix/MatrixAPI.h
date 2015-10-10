#ifndef OGLA_MATRIXAPI_H
#define OGLA_MATRIXAPI_H

#include "Matrix.h"
#include "CuCore.h"
#include "assert.h"

#ifdef CUDA

__hostdeviceinline__ void SetRe(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->reValues[c + r * m->columns] = v;
}

__hostdeviceinline__ floatt GetRe(math::Matrix* m, uintt c, uintt r) {
  return m->reValues[c + r * m->columns];
}

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->imValues[c + r * m->columns] = v;
}

__hostdeviceinline__ floatt GetIm(math::Matrix* m, uintt c, uintt r) {
  return m->imValues[c + r * m->columns];
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

__hostdeviceinline__ floatt GetRe(math::Matrix* m, uintt c, uintt r) {
  test::getRe(m, c, r, m->reValues[c + r * m->columns]);
  assert(c + r * m->columns < m->realRows * m->realColumns);
  return m->reValues[c + r * m->columns];
}

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->imValues[c + r * m->columns] = v;
  test::setIm(m, c, r, v);
  assert(c + r * m->columns < m->realRows * m->realColumns);
}

__hostdeviceinline__ floatt GetIm(math::Matrix* m, uintt c, uintt r) {
  test::getIm(m, c, r, m->imValues[c + r * m->columns]);
  assert(c + r * m->columns < m->realRows * m->realColumns);
  return m->imValues[c + r * m->columns];
}
__hostdeviceinline__ void Reset(math::Matrix* m) { test::reset(m); }

__hostdeviceinline__ void Push(math::Matrix* m) { test::push(m); }

__hostdeviceinline__ void Pop(math::Matrix* m) { test::pop(m); }

#endif

#endif  // MATRIXAPI_H
