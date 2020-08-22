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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "PatternsClassificationHost.h"

using namespace ::testing;

class OapNeuralRoutinesTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapNeuralRoutinesTests, PatternsClassification_ArgsParser_Test_1)
{
  oap::PatternsClassificationParser parser;

  constexpr int argc = 3;
  char* argv[argc] = {(char*)"application_name", (char*)"--pattern1", (char*)"test_path"};

  parser.parse (argc, argv);
  auto args = parser.getArgs ();

  EXPECT_EQ ("test_path", args.patternPath1);
  EXPECT_NE ("test_path", args.patternPath2);
}

TEST_F(OapNeuralRoutinesTests, PatternsClassification_ArgsParser_Test_2)
{
  oap::PatternsClassificationParser parser;

  constexpr int argc = 3;
  char* argv[argc] = {(char*)"application_name", (char*)"--pattern2", (char*)"test_path"};

  parser.parse (argc, argv);
  auto args = parser.getArgs ();

  EXPECT_NE ("test_path", args.patternPath1);
  EXPECT_EQ ("test_path", args.patternPath2);
}

TEST_F(OapNeuralRoutinesTests, PatternsClassification_ArgsParser_Test_3)
{
  oap::PatternsClassificationParser parser;

  constexpr int argc = 5;
  char* argv[argc] = {(char*)"application_name", (char*)"--pattern1", (char*)"test_path_1", (char*)"--pattern2", (char*)"test_path_2"};

  parser.parse (argc, argv);
  auto args = parser.getArgs ();

  EXPECT_EQ ("test_path_1", args.patternPath1);
  EXPECT_EQ ("test_path_2", args.patternPath2);
}

TEST_F(OapNeuralRoutinesTests, PatternsClassification_ArgsParser_Layers_Test)
{
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 3;
    char* argv[argc] = {(char*)"application_name", (char*)"--layers", (char*)"20,10,1"};

    parser.parse (argc, argv);
    auto args = parser.getArgs ();

    EXPECT_EQ(3, args.networkLayers.size());
    EXPECT_EQ(20, args.networkLayers[0]);
    EXPECT_EQ(10, args.networkLayers[1]);
    EXPECT_EQ(1, args.networkLayers[2]);
  }
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 3;
    char* argv[argc] = {(char*)"application_name", (char*)"--layers", (char*)"20,10,1,5"};

    parser.parse (argc, argv);
    auto args = parser.getArgs ();

    EXPECT_EQ(4, args.networkLayers.size());
    EXPECT_EQ(20, args.networkLayers[0]);
    EXPECT_EQ(10, args.networkLayers[1]);
    EXPECT_EQ(1, args.networkLayers[2]);
    EXPECT_EQ(5, args.networkLayers[3]);
  }
}

TEST_F(OapNeuralRoutinesTests, PatternsClassification_ArgsParser_ErrorType_Test)
{
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 7;
    char* argv[argc] = {(char*)"application_name",
                        (char*)"--pattern1", (char*)"test_path_1", 
                        (char*)"--pattern2", (char*)"test_path_2",
                        (char*)"--error_type", (char*)"cross_entropy"};

    parser.parse (argc, argv);
    auto args = parser.getArgs ();

    EXPECT_EQ ("test_path_1", args.patternPath1);
    EXPECT_EQ ("test_path_2", args.patternPath2);
    EXPECT_EQ (oap::ErrorType::CROSS_ENTROPY, args.errorType);
  }
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 7;
    char* argv[argc] = {(char*)"application_name",
                        (char*)"--pattern1", (char*)"test_path_1", 
                        (char*)"--pattern2", (char*)"test_path_2",
                        (char*)"--error_type", (char*)"ce"};

    parser.parse (argc, argv);
    auto args = parser.getArgs ();

    EXPECT_EQ ("test_path_1", args.patternPath1);
    EXPECT_EQ ("test_path_2", args.patternPath2);
    EXPECT_EQ (oap::ErrorType::CROSS_ENTROPY, args.errorType);
  }
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 7;
    char* argv[argc] = {(char*)"application_name",
                        (char*)"--pattern1", (char*)"test_path_1", 
                        (char*)"--pattern2", (char*)"test_path_2",
                        (char*)"--error_type", (char*)"mean_square_error"};

    parser.parse (argc, argv);
    auto args = parser.getArgs ();

    EXPECT_EQ ("test_path_1", args.patternPath1);
    EXPECT_EQ ("test_path_2", args.patternPath2);
    EXPECT_EQ (oap::ErrorType::MEAN_SQUARE_ERROR, args.errorType);
  }
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 7;
    char* argv[argc] = {(char*)"application_name",
                        (char*)"--pattern1", (char*)"test_path_1", 
                        (char*)"--pattern2", (char*)"test_path_2",
                        (char*)"--error_type", (char*)"mse"};

    parser.parse (argc, argv);
    auto args = parser.getArgs ();

    EXPECT_EQ ("test_path_1", args.patternPath1);
    EXPECT_EQ ("test_path_2", args.patternPath2);
    EXPECT_EQ (oap::ErrorType::MEAN_SQUARE_ERROR, args.errorType);
  }
  {
    oap::PatternsClassificationParser parser;

    constexpr int argc = 7;
    char* argv[argc] = {(char*)"application_name",
                        (char*)"--pattern1", (char*)"test_path_1", 
                        (char*)"--pattern2", (char*)"test_path_2",
                        (char*)"--error_type", (char*)"invalid_error_type"};

    EXPECT_THROW (parser.parse (argc, argv), std::exception);
  }
}
