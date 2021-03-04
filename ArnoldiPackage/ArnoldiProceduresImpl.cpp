/*
 * Copyright 2016 - 2021 Marcin Matula
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

CuHArnoldiCallbackBase::CuHArnoldiCallbackBase() : CuHArnoldi() {}

CuHArnoldiCallbackBase::~CuHArnoldiCallbackBase() {}

void CuHArnoldiDefault::multiply(math::ComplexMatrix* w, math::ComplexMatrix* v,
                                 oap::CuProceduresApi& cuProceduresApi,
                                 oap::VecMultiplicationType mt) {
  cuProceduresApi.dotProduct(w, m_A, v);
}

void CuHArnoldiCallbackBase::multiply(math::ComplexMatrix* w, math::ComplexMatrix* v, oap::CuProceduresApi& cuProceduresApi, oap::VecMultiplicationType mt)
{
  m_multiplyFunc(w, v, cuProceduresApi, m_userData, mt);
}

bool CuHArnoldiCallbackBase::checkEigenspair(floatt reevalue, floatt imevalue, math::ComplexMatrix* vector, uint index, uint max)
{
  return m_checkFunc(reevalue, imevalue, vector, index, max, m_checkUserData);
}

void CuHArnoldiCallbackBase::setCallback(CuHArnoldiCallbackBase::MultiplyFunc multiplyFunc, void* userData)
{
  m_multiplyFunc = multiplyFunc;
  m_userData = userData;
}

void CuHArnoldiCallbackBase::setCheckCallback(CuHArnoldiCallbackBase::CheckFunc checkFunc, void* userData)
{
  m_checkFunc = checkFunc;
  m_checkUserData = userData;
}

void CuHArnoldiCallback::execute (uint k, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType)
{
  CuHArnoldi::execute (k, wantedCount, matrixInfo, matrixType);
}

void CuHArnoldiCallbackThread::execute (uint k, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType)
{}

void CuHArnoldiCallbackThread::stop (const std::string& path)
{}

void CuHArnoldiCallbackThread::save (const std::string& path)
{}

void CuHArnoldiCallbackThread::load (const std::string& path)
{}
