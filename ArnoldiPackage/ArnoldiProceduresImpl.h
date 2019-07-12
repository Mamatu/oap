/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef ARNOLDIPROCEDURESIMPL_H
#define ARNOLDIPROCEDURESIMPL_H

#include "ArnoldiProcedures.h"
#include <thread>

class CuHArnoldiDefault : public CuHArnoldi {
 public:
  /**
 * @brief Set device matrix to calculate its eigenvalues and eigenvectors.
 * @param A
 */
  void setMatrix(math::Matrix* A) { m_A = A; }

 protected:
  virtual void multiply(math::Matrix* m_w, math::Matrix* m_v,
                        oap::CuProceduresApi& cuProceduresApi,
                        CuHArnoldi::MultiplicationType mt);

  virtual bool checkEigenspair(floatt reevalue, floatt imevalue, math::Matrix* vector, uint index, uint max) {
    return true;
  }

 private:
  math::Matrix* m_A;
};

class CuHArnoldiCallbackBase : public CuHArnoldi
{
  public:
    CuHArnoldiCallbackBase();
    virtual ~CuHArnoldiCallbackBase();

    typedef void (*MultiplyFunc)(math::Matrix* m_w, math::Matrix* m_v,  oap::CuProceduresApi& cuProceduresApi,
                                 void* userData, CuHArnoldi::MultiplicationType mt);

    typedef bool (*CheckFunc) (floatt reevalue, floatt imevalue, math::Matrix* vector, uint index, uint max, void* userData);

    void setCallback(MultiplyFunc multiplyFunc, void* userData);
    void setCheckCallback(CheckFunc multiplyFunc, void* userData);

  protected:
    virtual void multiply(math::Matrix* m_w, math::Matrix* m_v,
                          oap::CuProceduresApi& cuProceduresApi,
                          CuHArnoldi::MultiplicationType mt);

    virtual bool checkEigenspair(floatt reevalue, floatt imevalue, math::Matrix* vector, uint index, uint max);

 private:
  MultiplyFunc m_multiplyFunc;
  CheckFunc m_checkFunc;
  void* m_userData;
  void* m_checkUserData;
};

class CuHArnoldiCallback : public CuHArnoldiCallbackBase
{
  public:
    void execute (uint k, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType = ArnUtils::DEVICE);
};

class CuHArnoldiCallbackThread : public CuHArnoldiCallbackBase
{
  public:
  void execute (uint k, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType = ArnUtils::DEVICE);

  void stop (const std::string& path);

  void save (const std::string& path);
  void load (const std::string& path);
};

#endif  // ARNOLDIPROCEDURESIMPL_H
