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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "DataLoader.h"
#include "Image.h"

using namespace ::testing;

class OapPngLoaderTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:
  class PngFileMock : public oap::Image {
   public:
    PngFileMock() {}

    virtual ~PngFileMock() {}

    MOCK_METHOD1(openInternal, bool(const char*));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isPngInternal, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_METHOD0(close, void());

    MOCK_CONST_METHOD0(getWidth, size_t());

    MOCK_CONST_METHOD0(getHeight, size_t());

    MOCK_CONST_METHOD2(getPixelInternal, oap::pixel_t(unsigned int, unsigned int));

    MOCK_CONST_METHOD1(getPixelsVector, void(oap::pixel_t*));
  };
};

TEST_F(OapPngLoaderTests, LoadFail) {
  PngFileMock pngFileMock;

  EXPECT_CALL(pngFileMock, openInternal(_)).Times(1).WillOnce(Return(false));

  EXPECT_THROW(oap::DataLoader pngDataLoader(&pngFileMock, ""), oap::exceptions::FileNotExist);
}

TEST_F(OapPngLoaderTests, VerificationFail) {
  PngFileMock pngFileMock;

  EXPECT_CALL(pngFileMock, openInternal(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(pngFileMock, isPngInternal()).Times(1).WillOnce(Return(false));

  EXPECT_THROW(oap::DataLoader pngDataLoader(&pngFileMock, ""), oap::exceptions::FileIsNotPng);
}

TEST_F(OapPngLoaderTests, Load) {
  PngFileMock pngFileMock;

  EXPECT_CALL(pngFileMock, openInternal(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(pngFileMock, isPngInternal()).Times(1).WillOnce(Return(true));

  EXPECT_CALL(pngFileMock, loadBitmap()).Times(1);

  EXPECT_CALL(pngFileMock, freeBitmap()).Times(1);

  ON_CALL(pngFileMock, getWidth()).WillByDefault(Return(1024));

  ON_CALL(pngFileMock, getHeight()).WillByDefault(Return(760));

  EXPECT_CALL(pngFileMock, close()).Times(1);

  EXPECT_NO_THROW(oap::DataLoader pngDataLoader(&pngFileMock, ""));
}
