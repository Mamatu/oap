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


#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "DebugLogs.h"


class OapUtilsTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

void testFunction2() {
  traceFunction();
}

void testFunction1() {
  traceFunction();
  testFunction2();
}

void testFunction() {
  traceFunction();
  testFunction1();
}

TEST_F(OapUtilsTests, TraceLogTest) {
  initTraceBuffer(1024);
  testFunction();

  std::string buffer;
  getTraceOutput(buffer);

  debug("%s", buffer.c_str());

  std::size_t pos = buffer.find("testFunction");
  EXPECT_NE(pos, std::string::npos);

  pos = buffer.find("testFunction1");
  EXPECT_NE(pos, std::string::npos);

  pos = buffer.find("testFunction2");
  EXPECT_NE(pos, std::string::npos);
}
