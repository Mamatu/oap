/*
 * Copyright 2016 - 2018 Marcin Matula
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

class SquareErrorLimitController : public Network::IController
{
  size_t m_dataSetSize;
  size_t m_step;
  floatt m_limit;
  floatt m_sqes;
  bool m_sc;

  public:
   SquareErrorLimitController (floatt limit, size_t dataSetSize);
   virtual ~SquareErrorLimitController();

   virtual bool shouldCalculateError(size_t step) override;

   virtual void setSquareError (floatt sqe) override;

   virtual bool shouldContinue() override;
};

class DerivativeController : public Network::IController
{
  size_t m_activationLimit;
  public:
   DerivativeController (size_t activationLimit);
   virtual ~DerivativeController();

   virtual bool shouldCalculateError(size_t step) override;

   virtual void setSquareError (floatt sqe) override;

   virtual bool shouldContinue() override;
};

#endif
