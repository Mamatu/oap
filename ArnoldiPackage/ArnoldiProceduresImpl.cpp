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

#include "ArnoldiProceduresImpl.h"

CuHArnoldiCallback::CuHArnoldiCallback() : CuHArnoldi() {}

CuHArnoldiCallback::~CuHArnoldiCallback() {}

void CuHArnoldiDefault::multiply(math::Matrix* w, math::Matrix* v,
                                 CuHArnoldi::MultiplicationType mt) {
  m_cuMatrix.dotProduct(w, m_A, v);
}

void CuHArnoldiCallback::multiply(math::Matrix* w, math::Matrix* v,
                                  CuHArnoldi::MultiplicationType mt) {
  m_multiplyFunc(w, v, m_userData, mt);
}

bool CuHArnoldiCallback::checkEigenspair(floatt reevalue, floatt imevalue, math::Matrix* vector, uint index, uint max) {
  return m_checkFunc(reevalue, imevalue, vector, index, max, m_checkUserData);
}

void CuHArnoldiCallback::setCallback(CuHArnoldiCallback::MultiplyFunc multiplyFunc, void* userData) {
  m_multiplyFunc = multiplyFunc;
  m_userData = userData;
}

void CuHArnoldiCallback::setCheckCallback(CuHArnoldiCallback::CheckFunc checkFunc, void* userData) {
  m_checkFunc = checkFunc;
  m_checkUserData = userData;
}
