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

#ifndef PNGDATALOADER_H
#define PNGDATALOADER_H

#include "IPngFile.h"
#include "Exceptions.h"

namespace oap {

class PngDataLoader {
 public:
  PngDataLoader(IPngFile* ifile, const char* path);

  virtual ~PngDataLoader();

  Pixel getPixel(unsigned int x, unsigned int y) const;

 private:
  IPngFile* m_ifile;

  void load(const char* path);
};
}
#endif  // PNGLOADER_H
