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

#ifndef OAP_PATTERNS_CLASSIFICATION_H
#define OAP_PATTERNS_CLASSIFICATION_H

#include <atomic>
#include <string>
#include <functional>

#include "Routine.h"
#include "oapNetwork.h"
#include "PngFile.h"

#include "ThreadUtils.h"

#include "PatternsClassificationHost.h"

namespace oap
{

class PatternsClassification : public oap::Routine
{
  public:
    using Args = oap::PatternsClassificationParser::Args;

    PatternsClassification ();
    virtual ~PatternsClassification ();

    int run ();
    int run (const oap::PatternsClassificationParser::Args& args);

    virtual void onInterrupt() override;

  protected:
    virtual int runRoutine () override;
    virtual const oap::IArgsParser& getArgsParser() const override;

  private:
    oap::PatternsClassificationParser m_parser;
    std::atomic_bool m_bInterrupted;
    oap::utils::sync::CondBool m_cond;
};
}

#endif
