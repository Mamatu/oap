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

#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>

#include "Image.h"
#include "Exceptions.h"
#include "Math.h"

namespace oap {

class DataLoader {
 public:
  DataLoader(Image* ifile, const std::string& path, bool deallocateIFile = false);

  DataLoader(Image* ifile);

  template <typename T>
  static DataLoader* newDataLoader(const std::string& path) {
    return new DataLoader(new T(), path, true);
  }

  virtual ~DataLoader();

  oap::pixel_t getPixel(unsigned int x, unsigned int y) const;

  void getPixelsVector(oap::pixel_t* pixels) const;
  void getFloattVector(floatt* vector) const;

  size_t getWidth() const;
  size_t getHeight() const;
  size_t getLength() const;

 private:
  Image* m_ifile;
  bool m_deallocateIFile;

  void openAndLoad(const std::string& path);
  void load();
};
}
#endif  // PNGLOADER_H
