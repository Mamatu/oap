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
#include <pthread.h>
#include "gtest/gtest.h"
#include "TestProcedures.h"
#include "gmock/gmock-generated-function-mockers.h"

class oglaCudaTests : public testing::Test {
public:

    CuTest cuTest;
    CUresult status;

    virtual void SetUp() {
        cuda::Context::Instance().create();
        status = CUDA_SUCCESS;
    }

    virtual void TearDown() {
        cuda::Context::Instance().destroy();
    }
};

/*
TEST_F(oglaCudaTests, SyncBlocksAtomicTest) {
    bool c = cuTest.test2();
    EXPECT_EQ(true, c);
}

TEST_F(oglaCudaTests, SyncBlocksTest) {
    bool c = cuTest.test1();
    EXPECT_EQ(true, c);
}*/

