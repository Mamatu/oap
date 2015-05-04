// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Google Mock - a framework for writing C++ mock classes.
//
// This file tests code in gmock.cc.

#include <string>
#include <stdio.h>
#include <math.h>
#include "gtest/gtest.h"
#include "ThreadsMapper.h"

class OglaThreadsMapperTests : public testing::Test {
 public:
  int m_columns;
  int m_rows;
  int m_threadsLimit;
  int m_threadsLimitSqrt;

  uintt m_blocks[2];
  uintt m_threads[2];

  virtual void SetUp() {
    m_columns = -1;
    m_rows = -1;
    m_threadsLimit = 1024;
    m_threadsLimitSqrt = sqrt(m_threadsLimit);
    m_threads[0] = 0;
    m_threads[1] = 0;
    m_blocks[0] = 0;
    m_blocks[1] = 0;
  }

  virtual void TearDown() {}

  void execute() {
    utils::mapper::SetThreadsBlocks(m_blocks, m_threads, m_columns, m_rows,
                                    m_threadsLimit);

    EXPECT_GE(m_threadsLimit, m_threads[0] * m_threads[1]);
  }
};

TEST_F(OglaThreadsMapperTests, Test1) {
  m_columns = 20;
  m_rows = 20;
  execute();
  EXPECT_EQ(1, m_blocks[0]);
  EXPECT_EQ(1, m_blocks[1]);
  EXPECT_EQ(m_columns, m_threads[0]);
  EXPECT_EQ(m_rows, m_threads[1]);
}

TEST_F(OglaThreadsMapperTests, Test2) {
  m_columns = 256;
  m_rows = 256;
  execute();
  EXPECT_EQ(8, m_blocks[0]);
  EXPECT_EQ(8, m_blocks[1]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[0]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[1]);
  EXPECT_GE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_GE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OglaThreadsMapperTests, Test3) {
  m_columns = 1;
  m_rows = 16438;
  execute();
  EXPECT_EQ(1, m_blocks[0]);
  EXPECT_EQ(514, m_blocks[1]);
  EXPECT_EQ(1, m_threads[0]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OglaThreadsMapperTests, Test4) {
  m_columns = 16384;
  m_rows = 2;
  execute();
  //EXPECT_EQ(32, m_blocks[0]);
  //EXPECT_EQ(514, m_blocks[1]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[0]);
  EXPECT_EQ(2, m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}
