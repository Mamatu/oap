/*
 * Copyright 2016, 2017 Marcin Matula
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
  assert(c + r * m->columns < m->rows * m->columns);
}

__hostdeviceinline__ floatt GetRe(const math::Matrix* m, uintt c, uintt r) {
  test::getRe(m, c, r, m->reValues[c + r * m->columns]);
  assert(c + r * m->columns < m->rows * m->columns);
  return m->reValues[c + r * m->columns];
}

__hostdeviceinline__ floatt GetReIndex(const math::Matrix* m, uintt index) {
  test::getRe(m, index, m->reValues[index]);
  assert(index < m->rows * m->columns);
  return m->reValues[index];
}

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  m->imValues[c + r * m->columns] = v;
  test::setIm(m, c, r, v);
  assert(c + r * m->columns < m->rows * m->columns);
}

__hostdeviceinline__ floatt GetIm(const math::Matrix* m, uintt c, uintt r) {
  test::getIm(m, c, r, m->imValues[c + r * m->columns]);
  assert(c + r * m->columns < m->rows * m->columns);
  return m->imValues[c + r * m->columns];
}

__hostdeviceinline__ floatt GetImIndex(const math::Matrix* m, uintt index) {
  test::getIm(m, index, m->imValues[index]);
  assert(index < m->rows * m->columns);
  return m->imValues[index];
}

__hostdeviceinline__ void Reset(math::Matrix* m) {
  HOST_INIT();
  // threads_sync();
  test::reset(m);
  // threads_sync();
}

__hostdeviceinline__ void Push(math::Matrix* m) {
  HOST_INIT();
  // threads_sync();
  test::push(m);
  // threads_sync();
}

__hostdeviceinline__ void Pop(math::Matrix* m) {
  HOST_INIT();
  // threads_sync();
  test::pop(m);
  // threads_sync();
}

#endif

__hostdeviceinline__ uintt GetIndex(const math::Matrix* m, uintt c, uintt r) {
  return c + r * m->columns;
}

__hostdeviceinline__ floatt* GetRePtr(const math::Matrix* m, uintt c, uintt r) {
  return m->reValues + GetIndex(m, c, r);
}

__hostdeviceinline__ floatt* GetImPtr(const math::Matrix* m, uintt c, uintt r) {
  return m->imValues + GetIndex(m, c, r);
}

#endif  // MATRIXAPI_H
