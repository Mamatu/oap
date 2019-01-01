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

#ifndef OAP_HOST_KERNEL_EXECUTOR_H
#define OAP_HOST_KERNEL_EXECUTOR_H

#include "Dim3.h"
#include "IKernelExecutor.h"


class HostKernelExecutor : public oap::IKernelExecutor
{
  public:
    HostKernelExecutor();

    virtual ~HostKernelExecutor();

    virtual std::string getErrorMsg () const override;

    virtual uint getMaxThreadsPerBlock() const override;
    
  protected:
    virtual bool run(const char* functionName) override;
};

#endif
