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

#include "Config.h"
#include "DebugLogs.h"
#include "Exceptions.h"
#include "PngFile.h"

using namespace ::testing;

class OapPngFileTests : public testing::Test {
 public:
  OapPngFileTests() {
    m_data_path = utils::Config::getPathInOap("oap2dt3d/data/");
    m_images_path = m_data_path + "images_monkey/";
  }

  virtual void SetUp() {}

  virtual void TearDown() {}

  std::string m_data_path;
  std::string m_images_path;

  std::string getImagePath(const std::string& filename) {
    return m_images_path + filename;
  }
};

TEST_F(OapPngFileTests, WriteImageToFileTest) {
  EXPECT_NO_THROW(try {
    oap::PngFile pngFile(getImagePath("image000.png"));

    EXPECT_TRUE(oap::PngFile::writeImageToFile(&pngFile, "/tmp/o.png"));
  } catch (const oap::exceptions::Exception& ex) {
    debugException(ex);
    throw;
  });
}
