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

#include "ArnoldiProceduresImpl.h"

void CuHArnoldiDefault::multiply(math::Matrix* w, math::Matrix* v,
                                 CuHArnoldi::MultiplicationType mt) {
  m_cuMatrix.dotProduct(w, m_A, v);
}

void CuHArnoldiCallback::multiply(math::Matrix* w, math::Matrix* v,
                                  CuHArnoldi::MultiplicationType mt) {
  m_multiplyFunc(w, v, m_userData, mt);
}

void CuHArnoldiCallback::setCallback(
    CuHArnoldiCallback::MultiplyFunc multiplyFunc, void* userData) {
  m_multiplyFunc = multiplyFunc;
  m_userData = userData;
}