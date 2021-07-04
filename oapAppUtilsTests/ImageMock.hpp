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
#ifndef IMAGEMOCK_H
#define IMAGEMOCK_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "Image.hpp"

using namespace ::testing;

class ImageMock : public oap::Image {
 public:  // ImageMock type
  ImageMock(const std::string& path) : oap::Image(path) {}

  ImageMock() : oap::Image("test_path/test_file") {}

  virtual ~ImageMock() {}

  MOCK_METHOD1(openProtected, bool(const std::string& path));

  MOCK_METHOD3(read, bool(void*, size_t, size_t));

  MOCK_CONST_METHOD0(isCorrectFormat, bool());

  MOCK_METHOD0(loadBitmapProtected, void());

  MOCK_METHOD0(freeBitmapProtected, void());

  MOCK_METHOD0(closeProtected, void());

  MOCK_CONST_METHOD0(getWidth, oap::ImageSection());

  MOCK_CONST_METHOD0(getHeight, oap::ImageSection());

  MOCK_METHOD1(forceOutputWidth, void(const oap::ImageSection& optWidth));

  MOCK_METHOD1(forceOutputHeight, void(const oap::ImageSection& optHeight));

  MOCK_CONST_METHOD0(getOutputWidth, oap::ImageSection());

  MOCK_CONST_METHOD0(getOutputHeight, oap::ImageSection());

  MOCK_CONST_METHOD0(getSufix, std::string());

  MOCK_CONST_METHOD2(getPixelProtected,
                     oap::pixel_t(unsigned int, unsigned int));

  MOCK_CONST_METHOD1(getPixelsVectorProtected, void(oap::pixel_t*));

  MOCK_METHOD1(onSave, void(const std::string& path));
  MOCK_METHOD1(saveProtected, bool(const std::string& path));
};

using NiceImageMock = NiceMock<ImageMock>;

#endif  // IMAGEMOCK
