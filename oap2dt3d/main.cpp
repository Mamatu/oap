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

#include <stdlib.h>
#include <string>

#include "PngDataLoader.h"
#include "PngFile.h"

int main() {
  try {
    oap::PngFile pngFile;
    oap::PngDataLoader pngLoader(
        &pngFile,
        "/home/mmatula/Oap/oap2dt3d/data/images_monkey/image_0_0_0.png");
  } catch (const oap::exceptions::Exception& exception) {
    fprintf(stderr, "%s \n", exception.getMessage().c_str());
  }

  return 0;
}
