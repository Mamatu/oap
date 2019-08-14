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

#ifndef OAP_CONTROLLERS_H
#define OAP_CONTROLLERS_H

#include "oapNetwork.h"
#include <functional>
#include <queue>

/**
 * Square Error - Independent Data
 */
class SE_ID_Controller : public Network::IController
{
  uintt m_dataSetSize;
  uintt m_step;
  floatt m_limit;

  floatt m_sqes;
  bool m_sc;

  std::function<void(floatt, uintt, floatt)> m_callback;
  public:
   SE_ID_Controller (floatt limit, uintt dataSetSize, const std::function<void(floatt, uintt, floatt)>& callback = nullptr);
   virtual ~SE_ID_Controller();

   virtual bool shouldCalculateError(uintt step) override;

   virtual void setError (floatt sqe, oap::ErrorType etype) override;

   virtual bool shouldContinue() override;
};

/**
 * Square Error - Continous Data
 */
class SE_CD_Controller : public Network::IController
{
  uintt m_dataSetSize;
  uintt m_step;
  floatt m_limit;

  std::queue<floatt> m_sqes;
  floatt m_sqe;
  bool m_sc;

  std::function<void(floatt, uintt, floatt)> m_callback;
  public:
   SE_CD_Controller (floatt limit, uintt dataSetSize, const std::function<void(floatt, uintt, floatt)>& callback = nullptr);
   virtual ~SE_CD_Controller();

   virtual bool shouldCalculateError(uintt step) override;

   virtual void setError (floatt sqe, oap::ErrorType etype) override;

   virtual bool shouldContinue() override;
};

class DerivativeController : public Network::IController
{
  uintt m_activationLimit;
  public:
   DerivativeController (uintt activationLimit);
   virtual ~DerivativeController();

   virtual bool shouldCalculateError(uintt step) override;

   virtual void setError (floatt sqe, oap::ErrorType etype) override;

   virtual bool shouldContinue() override;
};

class StepController : public Network::IController
{
  uintt m_sstep;
  uintt m_step = 0;
  public:
    StepController (uintt sstep);
    virtual ~StepController ();

    virtual bool shouldCalculateError(uintt step) override;

   virtual void setError (floatt sqe, oap::ErrorType etype) override;

   virtual bool shouldContinue() override;
};

#endif
